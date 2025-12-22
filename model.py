import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from sklearn.metrics import fbeta_score

# -----------------------------------------------------------------------------
# Core Model Components

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

def init_linear(w: Tensor):
    std = 0.5 * (w.size(-1) ** -0.5)
    bound = (3 ** 0.5) * std
    return w.uniform_(-bound, bound)

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

# -----------------------------------------------------------------------------
# Batched Attention (replaces FlexAttention)

class CausalSelfAttentionBatched(nn.Module):
    """Standard batched causal attention - no FlexAttention constraints
    """
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        self.qkvo_w = nn.Parameter(torch.empty(4, hdim, dim).bfloat16())
        self.rotary = Rotary(head_dim, max_seq_len)
        self.attn_scale = 0.12
        self.max_seq_len = max_seq_len

    def forward(self, x: Tensor, ve: Tensor | None, sa_lambdas: Tensor):
        """Forward pass with value embedding support
        
        Args:
            x: Input tensor [B, T, D]
            ve: Value embeddings [B, T, D] or None
            sa_lambdas: Self-attention lambdas [2] for mixing v with ve
        """
        B, T, D = x.shape
        
        qkv = F.linear(x, self.qkvo_w[:3].flatten(end_dim=1)).view(B, T, 3 * self.num_heads, self.head_dim)
        q, k, v = qkv.chunk(3, dim=-2)
        
        # Apply normalization and rotary embeddings
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        v = norm(v)
        
        if ve is not None:
            v = sa_lambdas[0] * v + sa_lambdas[1] * ve.view_as(v)
        else:
            v = sa_lambdas[0] * v
        
        # Reshape for attention: [B, num_heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.attn_scale
        
        # Apply causal mask
        if not hasattr(self, '_causal_mask') or self._causal_mask.size(0) < T:
            # Create and cache causal mask
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            self.register_buffer('_causal_mask', causal_mask, persistent=False)
        
        attn = attn.masked_fill(self._causal_mask[:T, :T], float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        y = attn @ v  # [B, num_heads, T, head_dim]
        
        # Reshape back and apply output projection
        y = y.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, self.qkvo_w[3])
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc_w = nn.Parameter(init_linear(torch.empty(hdim, dim)).bfloat16())
        self.proj_w = nn.Parameter(torch.zeros(dim, hdim).bfloat16())
        self.fc_w.wd_mul = 2.0
        self.proj_w.wd_mul = 2.0

    def forward(self, x: Tensor):
        x = F.linear(x, self.fc_w)
        x = F.relu(x).square()
        x = F.linear(x, self.proj_w)
        return x

class BlockBatched(nn.Module):
    """Block using standard batched attention"""
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttentionBatched(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, lambdas: Tensor, sa_lambdas: Tensor):
        """Forward pass matching original Block signature
        
        Args:
            x: Input tensor [B, T, D]
            ve: Value embeddings [B, T, D] or None
            x0: Initial embeddings [B, T, D]
            lambdas: Block lambdas [2] for residual scaling
            sa_lambdas: Self-attention lambdas [2] for value mixing
        """
        # Apply residual scaling
        x = (lambdas[0] * x + lambdas[1] * x0).type_as(x)
        
        # Self-attention (if not skipped)
        if self.attn is not None:
            x = x + self.attn(x, ve, sa_lambdas)
        
        # MLP
        x = x + self.mlp(norm(x))
        return x

# -----------------------------------------------------------------------------
# Batched GPT Model

class GPTBatched(nn.Module):
    """Batched version of GPT for classification fine-tuning
    
    This model removes FlexAttention constraints and supports arbitrary batch sizes.
    It can load weights from a pretrained FlexAttention GPT model.
    """
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        
        # Initialize embeddings
        self.embed = nn.Embedding(vocab_size, model_dim)
        
        self.value_embeds = nn.ModuleList([
            nn.Embedding(vocab_size, model_dim) for _ in range(3)
        ])
        
        # Create blocks
        self.blocks = nn.ModuleList([
            BlockBatched(model_dim, num_heads, max_seq_len, i) 
            for i in range(num_layers)
        ])
        
        # Learnable scalars for skip connections, residual scaling, AND self-attention mixing
        assert num_layers % 2 == 0
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers),  # skip_weights
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)],  # block lambdas
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)],
        ]))
    
    @staticmethod
    def from_pretrained_gpt(original_gpt):
        """Create a batched GPT from a FlexAttention GPT model

        Args:
            original_gpt: A GPT model trained with FlexAttention (from training script)

        Returns:
            GPTBatched model with copied weights
        """
        # Get architecture parameters
        model_dim = original_gpt.embed.embedding_dim
        vocab_size = original_gpt.embed.num_embeddings
        num_layers = len(original_gpt.blocks)

        # Infer num_heads from attention layer
        first_attn = next(block.attn for block in original_gpt.blocks if block.attn is not None)
        num_heads = first_attn.num_heads

        # Get max_seq_len from rotary embeddings
        max_seq_len = first_attn.rotary.cos.size(0)

        # Create new batched model
        batched_model = GPTBatched(vocab_size, num_layers, num_heads, model_dim, max_seq_len)

        # Copy embedding weights
        batched_model.embed.weight.data.copy_(original_gpt.embed.weight.data)

        for i, (old_ve, new_ve) in enumerate(zip(original_gpt.value_embeds, batched_model.value_embeds)):
            new_ve.weight.data.copy_(old_ve.weight.data)

        # Copy block weights - need to convert from training script format
        for i, (old_block, new_block) in enumerate(zip(original_gpt.blocks, batched_model.blocks)):
            # Copy attention weights if exists
            if old_block.attn is not None and new_block.attn is not None:
                # Convert qkv_w + c_proj.weight -> qkvo_w
                # old_block.attn.qkv_w: [3, hdim, dim]
                # old_block.attn.c_proj.weight: [dim, hdim]
                # new_block.attn.qkvo_w: [4, hdim, dim]

                qkv = old_block.attn.qkv_w.data  # [3, hdim, dim]
                o_proj = old_block.attn.c_proj.weight.data.T  # [hdim, dim]

                # Stack them together
                new_block.attn.qkvo_w.data[0].copy_(qkv[0])  # Q
                new_block.attn.qkvo_w.data[1].copy_(qkv[1])  # K
                new_block.attn.qkvo_w.data[2].copy_(qkv[2])  # V
                new_block.attn.qkvo_w.data[3].copy_(o_proj)  # O

            # Copy MLP weights - convert from layers to parameters
            # old: c_fc.weight [hdim, dim], c_proj.weight [dim, hdim]
            # new: fc_w [hdim, dim], proj_w [dim, hdim]
            new_block.mlp.fc_w.data.copy_(old_block.mlp.c_fc.weight.data)
            new_block.mlp.proj_w.data.copy_(old_block.mlp.c_proj.weight.data)

        # Copy scalars, handling potential size mismatch from distributed training padding
        min_size = min(batched_model.scalars.size(0), original_gpt.scalars.size(0))
        batched_model.scalars.data[:min_size].copy_(original_gpt.scalars.data[:min_size])

        return batched_model
    
    def forward_features(self, input_seq: Tensor):
        """Extract features from the model
        
        Args:
            input_seq: Input token IDs [B, T]
            
        Returns:
            Features tensor [B, T, model_dim]
        """
        B, T = input_seq.shape
        
        # Embed tokens
        x = x0 = norm(self.embed(input_seq))
        
        # Pattern: layers 0,1,2 get ve[0,1,2], middle layers get None, layers 13,14,15 get ve[0,1,2]
        ve_raw = [value_embed(input_seq) for value_embed in self.value_embeds]
        ve = [ve_raw[0], ve_raw[1], ve_raw[2]] + [None] * (len(self.blocks) - 6) + [ve_raw[0], ve_raw[1], ve_raw[2]]
        assert len(ve) == len(self.blocks)
        
        # Process through blocks with skip connections
        skip_connections = []
        skip_map = {9: 6, 10: 4, 11: 2}
        
        skip_weights = self.scalars[:len(self.blocks)]
        lambdas = self.scalars[len(self.blocks):3*len(self.blocks)].view(-1, 2)
        sa_lambdas = self.scalars[3*len(self.blocks):5*len(self.blocks)].view(-1, 2)
        
        for i, block in enumerate(self.blocks):
            # Add skip connection if applicable
            if i in skip_map:
                x = (x + skip_weights[skip_map[i]] * skip_connections[skip_map[i]]).type_as(x)
            
            # Apply block with value embeddings
            x = block(x, ve[i], x0, lambdas[i], sa_lambdas[i])
            skip_connections.append(x)
        
        # Final normalization
        x = norm(x)
        return x

# -----------------------------------------------------------------------------
# Classification Model Wrapper

class GPTClassification(nn.Module):
    """Wrapper around pretrained GPT for classification tasks
    
    This version uses batched attention for much faster training.
    
    Args:
        device: Device to run model on
        vocab_file: Path to tokenizer vocab file
        merges_file: Path to tokenizer merges file (can be None for GPT-2)
        data_dir: Directory to save/load processed data
        num_classes: Number of output classes (default: 2 for binary classification)
        num_embed: Model embedding dimension (must match pretrained model)
        dropout: Dropout rate for classification head
        context_size: Maximum sequence length
        batch_size: Batch size for training (default: 16)
        ipa: Whether to use IPA (unused, for compatibility)
        text_column: Name of text column(s) in dataset - can be single string or list of strings
        label_column: Name of label column in dataset (default: 'label')
    """
    
    def __init__(self, device, vocab_file, merges_file, data_dir, num_classes=2, 
                 num_embed=1024, dropout=0.1, context_size=1024, batch_size=16, ipa=False,
                 text_column='sentence', label_column='label'):
        super().__init__()
        
        # Validate context size is multiple of 128 (for compatibility, though not strictly required)
        if context_size % 128 != 0:
            print(f"Warning: context_size {context_size} is not a multiple of 128. Rounding up.")
            context_size = ((context_size // 128) + 1) * 128
        
        self.device = device
        self.num_classes = num_classes
        self.context_window = context_size
        self.batch_size = batch_size  # Now supports batch_size > 1!
        self.dropout_rate = dropout
        
        # Column names for dataset
        self.text_column = text_column if isinstance(text_column, list) else [text_column]
        self.label_column = label_column
        
        # Training hyperparameters
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.grad_clip = 1.0
        self.warmup_iter_ratio = 0.1
        self.lr_decay_iter_ratio = 0.9
        self.min_lr = 1e-5
        
        # Tokenizer
        self.tokenizer = Tokenizer.from_file(str(vocab_file))
        
        # Data paths
        self.data_dir = Path(data_dir)
        self.train_data_path = self.data_dir / "train.bin"
        self.val_data_path = self.data_dir / "val.bin"
        
        # Data will be loaded after prepare_if_needed() is called
        self.train_data = None
        self.train_labels = None
        self.val_data = None
        self.val_labels = None

        # Pad token ID (used for masking padding tokens)
        self.pad_token_id = 0  # Tokenizer pads with 0

        # Pretrained model (to be loaded)
        self.pretrained_model = None

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_embed, num_embed // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_embed // 2, num_classes)
        )

        # Class weights for handling imbalanced datasets (will be computed from training data)
        self.class_weights = None
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass for classification

        Args:
            input_ids: Input token IDs [B, T]
            attention_mask: Attention mask [B, T] (optional). 1 for real tokens, 0 for padding.
                          If None, computed from input_ids using pad_token_id.
            labels: Labels [B] (optional)

        Returns:
            If labels provided: (logits, loss)
            Otherwise: logits
        """
        # Get features from pretrained model
        features = self.pretrained_model.forward_features(input_ids)  # [B, T, D]

        # Compute attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()

        # Find the last non-padding token for each sequence
        # This handles both left and right padding by taking the rightmost non-pad token
        batch_size = input_ids.shape[0]
        token_indices = torch.arange(input_ids.shape[1], device=input_ids.device)
        last_non_pad_indices = (token_indices * attention_mask).argmax(dim=1)

        # Pool using the last non-padding token (standard for causal models)
        pooled = features[torch.arange(batch_size, device=features.device), last_non_pad_indices]

        # Classification head
        logits = self.classifier(pooled)  # [B, num_classes]

        if labels is not None:
            # Use class weights if available to handle imbalanced datasets
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
            return logits, loss
        return logits
    
    def prepare_if_needed(self, train_dataset, val_dataset, force_tokenization=False):
        """Prepare binary data files for training"""
        if force_tokenization or not self.train_data_path.exists():
            print(f"Tokenizing training data...")
            self._tokenize_and_save(train_dataset, self.train_data_path)
        
        if force_tokenization or not self.val_data_path.exists():
            print(f"Tokenizing validation data...")
            self._tokenize_and_save(val_dataset, self.val_data_path)

        # Load data once into memory
        print(f"Loading training data into memory...")
        self._load_data_into_memory()

        # Compute class weights to handle imbalanced datasets
        self._compute_class_weights()
    
    def _load_data_into_memory(self):
        """Load all data into memory once"""
        print(f"Loading train data from {self.train_data_path}...")
        with open(self.train_data_path, 'rb') as f:
            num_examples = np.fromfile(f, dtype=np.int32, count=1)[0]
            self.train_data = np.fromfile(f, dtype=np.int32,
                                         count=num_examples * self.context_window).reshape(num_examples, self.context_window)
            self.train_labels = np.fromfile(f, dtype=np.int64, count=num_examples)
        print(f"Loaded {len(self.train_data)} training examples")

        print(f"Loading val data from {self.val_data_path}...")
        with open(self.val_data_path, 'rb') as f:
            num_examples = np.fromfile(f, dtype=np.int32, count=1)[0]
            self.val_data = np.fromfile(f, dtype=np.int32,
                                       count=num_examples * self.context_window).reshape(num_examples, self.context_window)
            self.val_labels = np.fromfile(f, dtype=np.int64, count=num_examples)
        print(f"Loaded {len(self.val_data)} validation examples")

    def _compute_class_weights(self):
        """Compute class weights from training data to handle imbalanced datasets

        Uses inverse frequency weighting: weight_i = total_samples / (num_classes * count_i)
        This ensures minority classes get higher weights in the loss function.
        """
        if self.train_labels is None:
            print("Warning: Cannot compute class weights - training labels not loaded")
            return

        # Count samples per class
        unique_labels, counts = np.unique(self.train_labels, return_counts=True)
        print(f"\nClass distribution in training data:")
        for label, count in zip(unique_labels, counts):
            percentage = 100.0 * count / len(self.train_labels)
            print(f"  Class {label}: {count:,} samples ({percentage:.2f}%)")

        # Compute inverse frequency weights
        total_samples = len(self.train_labels)
        weights = np.zeros(self.num_classes, dtype=np.float32)

        for label, count in zip(unique_labels, counts):
            weights[label] = total_samples / (self.num_classes * count)

        # Convert to tensor and move to device
        self.class_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        print(f"\nComputed class weights (for balanced loss):")
        for i, weight in enumerate(weights):
            print(f"  Class {i}: {weight:.4f}")
        print()
    
    def _tokenize_and_save(self, dataset, save_path):
        """Tokenize dataset and save to binary file
        
        Supports both single column and multiple columns.
        For multiple columns, they are concatenated with ' <ENDOFTEXT> ' separator.
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        all_input_ids = []
        all_labels = []
        
        for example in dataset:
            # Handle single or multiple text columns
            if len(self.text_column) == 1:
                text = str(example[self.text_column[0]])
            else:
                # Concatenate multiple columns with separator
                text = ' <ENDOFTEXT> '.join([str(example[col]) for col in self.text_column])
            
            # Tokenize text
            encoded = self.tokenizer.encode(text)
            input_ids = encoded.ids[:self.context_window]
            
            # Pad if necessary
            if len(input_ids) < self.context_window:
                input_ids = input_ids + [0] * (self.context_window - len(input_ids))
            
            all_input_ids.append(input_ids)
            all_labels.append(example[self.label_column])
        
        # Convert to numpy arrays
        input_ids_array = np.array(all_input_ids, dtype=np.int32)
        labels_array = np.array(all_labels, dtype=np.int64)
        
        # Save as binary file
        with open(save_path, 'wb') as f:
            num_examples = np.array([len(all_input_ids)], dtype=np.int32)
            num_examples.tofile(f)
            input_ids_array.tofile(f)
            labels_array.tofile(f)
        
        print(f"Saved {len(all_input_ids)} examples to {save_path}")
        print(f"Text columns used: {self.text_column}")
    
    def get_batch(self, split):
        """Get a batch of data - now returns multiple examples!
        
        Args:
            split: 'train' or 'val'
            
        Returns:
            input_ids: [batch_size, context_window]
            labels: [batch_size]
        """
        if split == 'train':
            if self.train_data is None:
                raise RuntimeError("Training data not loaded! Call prepare_if_needed() first.")
            input_ids = self.train_data
            labels = self.train_labels
        else:
            if self.val_data is None:
                raise RuntimeError("Validation data not loaded! Call prepare_if_needed() first.")
            input_ids = self.val_data
            labels = self.val_labels
        
        # Sample batch_size examples
        indices = np.random.randint(0, len(input_ids), size=self.batch_size)
        input_ids_batch = torch.from_numpy(input_ids[indices]).to(self.device, dtype=torch.int32)
        labels_batch = torch.from_numpy(labels[indices]).to(self.device, dtype=torch.int64)
        
        return input_ids_batch, labels_batch
    
    def estimate_loss(self, ctx, eval_iters):
        """Estimate loss on train and val sets"""
        out = {}
        self.eval()

        for split in ['train', 'val']:
            losses = []
            accuracies = []
            f2_scores = []

            for _ in range(eval_iters):
                with torch.no_grad():
                    with ctx:
                        X, Y = self.get_batch(split)
                        attention_mask = (X != self.pad_token_id).long()
                        logits, loss = self(X, attention_mask=attention_mask, labels=Y)
                        
                        # Calculate accuracy
                        preds = logits.argmax(dim=-1)
                        acc = (preds == Y).float().mean().item()
                        
                        # Calculate F2 score
                        f2 = self._calculate_f2(preds, Y)
                        
                        losses.append(loss.item())
                        accuracies.append(acc)
                        f2_scores.append(f2)
            
            out[split] = np.mean(losses)
            out[f'{split}_accuracy'] = np.mean(accuracies)
            out[f'{split}_f2'] = np.mean(f2_scores)
        
        self.train()
        return out
    
    def _calculate_f2(self, preds, labels, beta=2):
        """Calculate F-beta score for multi-class classification
        
        Uses macro averaging to treat all classes equally.
        Works for both binary and multi-class classification.
        """
        preds_np = preds.cpu().numpy().flatten()
        labels_np = labels.cpu().numpy().flatten()
        
        try:
            f_beta = fbeta_score(labels_np, preds_np, beta=beta, average='macro', zero_division=0)
            return float(f_beta)
        except Exception:
            return 0.0
    
    def get_token_count(self):
        """Get total number of tokens in training set"""
        if self.train_data is None:
            with open(self.train_data_path, 'rb') as f:
                num_examples = np.fromfile(f, dtype=np.int32, count=1)[0]
            return num_examples * self.context_window
        return len(self.train_data) * self.context_window
    
    def get_metadata(self):
        """Get metadata about the tokenizer"""
        return {
            'vocab_size': self.tokenizer.get_vocab_size(),
            'max_length': self.context_window
        }


# -----------------------------------------------------------------------------
# Helper classes to load training script checkpoints

class CastedLinear(nn.Module):
    """Minimal implementation of CastedLinear for loading checkpoints"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

    def forward(self, x: Tensor):
        return F.linear(x, self.weight)

class TrainingScriptAttention(nn.Module):
    """Attention module matching the training script format"""
    def __init__(self, dim: int, num_heads: int, max_seq_len: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = 128
        hdim = num_heads * 128
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim))
        self.c_proj = CastedLinear(hdim, dim)
        self.rotary = Rotary(128, max_seq_len)

class TrainingScriptMLP(nn.Module):
    """MLP module matching the training script format"""
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)

class TrainingScriptBlock(nn.Module):
    """Block matching the training script format"""
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        self.attn = TrainingScriptAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = TrainingScriptMLP(dim)

class GPTFlexAttention(nn.Module):
    """Model structure that matches training script checkpoints

    This is used to load checkpoints from train_gpt_small.py and train_gpt_medium.py
    """
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([
            TrainingScriptBlock(model_dim, num_heads, max_seq_len, i)
            for i in range(num_layers)
        ])

        # The training script pads vocab_size to next multiple of 128
        padded_vocab_size = ((vocab_size // 128) + 1) * 128
        self.lm_head = CastedLinear(model_dim, padded_vocab_size)

        # Scalars may have padding for distributed training
        # We'll handle variable sizes when loading
        self.scalars = nn.Parameter(torch.zeros(5 * num_layers))