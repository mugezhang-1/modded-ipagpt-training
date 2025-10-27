"""
Generic dataset preprocessing script for headered .bin format
Supports any HuggingFace dataset and custom tokenizers
"""
import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

def write_datafile(filename, toks):
    """ 
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as a uint16
    """
    assert len(toks) < 2**31, "token count too large" # ~2.1B tokens
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    header[1] = 1 # version
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    # construct the tokens numpy array, if not already
    if not isinstance(toks, np.ndarray) or not toks.dtype == np.uint16:
        # validate that no token exceeds a uint16
        maxtok = 2**16
        assert all(0 <= t < maxtok for t in toks), "token dictionary too large for uint16"
        toks_np = np.array(toks, dtype=np.uint16)
    else:
        toks_np = toks
    # write to file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

# ------------------------------------------

parser = argparse.ArgumentParser(description="Generic dataset preprocessing for headered .bin format")
parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset name (e.g., 'HuggingFaceFW/fineweb')")
parser.add_argument("--dataset_config", type=str, default=None, help="Dataset configuration/subset name (e.g., 'sample-10BT')")
parser.add_argument("--split", type=str, default="train", help="Dataset split to use (default: 'train')")
parser.add_argument("--text_field", type=str, default="text", help="Field name containing text data (default: 'text')")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .bin files")
parser.add_argument("--shard_size", type=int, default=10**8, help="Size of each shard in tokens (default: 100M)")
parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer to use: 'gpt2', 'cl100k_base', 'o200k_base', or path to custom tokenizer file")
parser.add_argument("--eot_token", type=int, default=None, help="End-of-text token ID (auto-detected if not specified)")
args = parser.parse_args()

# create the output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# download/load the dataset
print(f"Loading dataset: {args.dataset}")
if args.dataset_config:
    print(f"  Config: {args.dataset_config}")
    dataset = load_dataset(args.dataset, name=args.dataset_config, split=args.split)
else:
    dataset = load_dataset(args.dataset, split=args.split)
print(f"  Split: {args.split}")
print(f"  Size: {len(dataset)} documents")

# initialize the tokenizer
print(f"Loading tokenizer: {args.tokenizer}")
if args.tokenizer in ["gpt2", "cl100k_base", "o200k_base"]:
    # Use tiktoken pre-trained tokenizer
    enc = tiktoken.get_encoding(args.tokenizer)
    eot = enc._special_tokens['<|endoftext|>'] if args.eot_token is None else args.eot_token
    
    def tokenize(doc):
        tokens = [eot]
        tokens.extend(enc.encode_ordinary(doc[args.text_field]))
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
        tokens_np_uint16 = tokens_np.astype(np.uint16)
        return tokens_np_uint16
    
else:
    # Load custom tokenizer from file
    try:
        from tokenizers import Tokenizer
        enc = Tokenizer.from_file(args.tokenizer)
        vocab_size = enc.get_vocab_size()
        print(f"  Custom tokenizer loaded with vocab size: {vocab_size}")
        
        # Try to detect EOT token
        if args.eot_token is None:
            # Common EOT token names
            for token_name in ['<|endoftext|>', '[EOS]', '</s>', '<eos>']:
                eot = enc.token_to_id(token_name)
                if eot is not None:
                    print(f"  Auto-detected EOT token: '{token_name}' (ID: {eot})")
                    break
            if eot is None:
                print("  Warning: Could not auto-detect EOT token. Using ID 0. Specify with --eot_token if needed.")
                eot = 0
        else:
            eot = args.eot_token
        
        def tokenize(doc):
            tokens = [eot]
            encoded = enc.encode(doc[args.text_field])
            tokens.extend(encoded.ids)
            tokens_np = np.array(tokens)
            assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
            tokens_np_uint16 = tokens_np.astype(np.uint16)
            return tokens_np_uint16
            
    except ImportError:
        print("Error: Custom tokenizer specified but 'tokenizers' library not installed.")
        print("Install with: pip install tokenizers")
        exit(1)
    except Exception as e:
        print(f"Error loading tokenizer from {args.tokenizer}: {e}")
        exit(1)

print(f"Using EOT token ID: {eot}")
print(f"Text field: '{args.text_field}'")
print(f"Shard size: {args.shard_size:,} tokens")
print(f"Output directory: {args.output_dir}")

# tokenize all documents and write output shards
nprocs = max(1, os.cpu_count() - 2)
print(f"Using {nprocs} processes for tokenization")

with mp.Pool(nprocs) as pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((args.shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    
    for tokens in pool.imap(tokenize, dataset, chunksize=16):
        # enough space in the current shard for the new tokens?
        if token_count + len(tokens) < args.shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(args.output_dir, f"data_{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = args.shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(args.output_dir, f"data_{split}_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens_np[:token_count])

print(f"\nPreprocessing complete! Created {shard_index + 1} shard(s) in {args.output_dir}")