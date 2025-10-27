# Modded NanoGPT Custom Training Pipeline

A modified NanoGPT implementation optimized for custom tokenizer pretraining and classification fine-tuning.

## Pipeline Summary

The training pipeline follows these steps:

**Step 1:** Clone this repository  
**Step 2:** Set up the conda environment  
**Step 3:** Preprocess your HuggingFace dataset with a custom tokenizer (generates .bin files)  
**Step 4:** Pretrain the GPT model using the .bin files (requires 8 GPUs)  
**Step 5:** Fine-tune the pretrained model on a classification task (e.g., XNLI)  
**Step 6:** Evaluate the fine-tuned model on test data  

## Step 1: Clone Repository

Clone this repository:

```bash
git clone https://github.com/mugezhang-1/modded-ipagpt-training.git
cd modded-ipagpt-training
```

## Step 2: Environment Setup

Create a conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate ipagpt_training
```

## Step 3: Data Preprocessing

Preprocess your HuggingFace dataset using a custom tokenizer to generate the required .bin files for training.

### Usage

```bash
python preprocessor.py   --dataset <HUGGINGFACE_DATASET_NAME>   --split train   --text_field <TEXT_FIELD_NAME>   --output_dir <OUTPUT_DIRECTORY>   --tokenizer <PATH_TO_TOKENIZER_JSON>   --shard_size 100000000
```

### Example

```bash
python preprocessor.py   --dataset "mugezhang/eng_spa_owt_word_full"   --split train   --text_field text   --output_dir "/fs/scratch/PAS2836/mugezhang/ipa_gpt_data/tokenized_bin/tokenized_bin_eng_spa_phonemized"   --tokenizer "/users/PAS2836/mugezhang/projects/ipa-gpt/train_tokenizer/tokenizers/eng_spa_owt/bpe-eng-spa-tokenizer.json"   --shard_size 100000000
```


### Key Arguments

| Argument          | Required | Description                                                             | Example                              |
|-------------------|----------|-------------------------------------------------------------------------|--------------------------------------|
| `--dataset`       | Yes      | Hugging Face dataset name                                               | `HuggingFaceFW/fineweb`              |
| `--dataset_config`| No       | Dataset configuration/subset (for datasets with multiple configs)       | `sample-10BT`                        |
| `--split`         | No       | Dataset split to use (default: `train`)                                 | `train`                              |
| `--text_field`    | No       | Field containing text data (default: `text`)                            | `text`                               |
| `--output_dir`    | Yes      | Directory for output `.bin` files                                       | `./data/my_dataset`                  |
| `--tokenizer`     | No       | Path to custom tokenizer JSON file (default: `gpt2`)                    | `./tokenizers/my_tokenizer.json`     |
| `--shard_size`    | No       | Tokens per shard (default: `100000000`)                                 | `100000000`                          |
| `--eot_token`     | No       | End-of-text token ID (auto-detected if not specified)                   | `50256`                              |


### Output

The script generates sharded `.bin` files in the specified output directory, for example:

- `data_val_000000.bin` (first shard, used for validation)
- `data_train_000001.bin`, `data_train_000002.bin`, ...

Each `.bin` file contains:

- A 256-int32 header with metadata (magic number, version, token count, etc.)
- Tokenized data as `uint16` values

> **Note:** Keep these file paths handy for the training step (Step 4).

## Step 4: Pretraining

Train the GPT model from scratch using your preprocessed `.bin` files on an 8-GPU setup via SLURM.

---

### 4.1: Calculate Token Count

First, determine how many training iterations you need by counting tokens in your dataset:

```bash
python count_tokens.py --data_dir /path/to/data --epochs 3 --tokens_per_iter 524288
```

The script will output:

- Token count per training file  
- Total training tokens  
- Iterations for specified epochs (based on 524,288 tokens per iteration for 8 GPUs, which is the default in the pretraining script)

> **Note:** We recommend training for at least **3 epochs**.

---

### 4.2: Create SLURM Script

Create a SLURM batch script (e.g., `train_job.slurm`) to launch distributed training:

```bash
#!/bin/bash
#SBATCH --job-name=modgpt-8gpu
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "Start: $(date)"
echo "Job $SLURM_JOB_ID on $SLURM_JOB_NODELIST"
cd "/path/to/modded-nanogpt"

# Activate conda environment
module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ipagpt_training

# Set communication and threading variables
export OMP_NUM_THREADS=4
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="^lo,docker0"

# Dataset paths (from Step 3)
DATA_DIR="/path/to/your/tokenized/data"
TRAIN_FILES="${DATA_DIR}/data_train_*.bin"
VAL_FILES="${DATA_DIR}/data_val_*.bin"

# Distributed training setup
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
MASTER_PORT=$((10000 + SLURM_JOB_ID % 20000))
NNODES=$SLURM_JOB_NUM_NODES
NODE_RANK=$SLURM_NODEID

# Launch distributed training
srun --ntasks-per-node=1 python -m torch.distributed.run   --nnodes="${NNODES}"   --nproc-per-node=4   --node_rank="${NODE_RANK}"   --rdzv_backend=c10d   --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"   train_gpt_medium.py   --train_files "${TRAIN_FILES}"   --val_files "${VAL_FILES}"   --num_iterations 50083   --val_loss_every 500   --vocab_size 50000   --output_dir "/fs/scratch/PAS2836/mugezhang/ipa_gpt_models/checkpoints/normal_eng_spa"   --save_checkpoint

echo "End: $(date)"
```

---

### 4.3: Key Training Arguments

| Argument | Required | Description | Example |
|-----------|-----------|-------------|----------|
| `--train_files` | Yes | Pattern for training `.bin` files | `./data/my_data/data_train_*.bin` |
| `--val_files` | Yes | Pattern for validation `.bin` files | `./data/my_data/data_val_*.bin` |
| `--num_iterations` | No | Number of training iterations (calculate from Step 4.1) | `50083` |
| `--val_loss_every` | No | Validation frequency in iterations (default: 125) | `500` |
| `--vocab_size` | No | Vocabulary size matching your tokenizer (default: 50257) | `50000` |
| `--output_dir` | No | Directory to save checkpoints (default: logs) | `./checkpoints` |
| `--save_checkpoint` | No | Flag to save model checkpoints | *(no value needed)* |

**Important:**

- The `--vocab_size` must match your custom tokenizer's vocabulary size  
- The `--num_iterations` should be calculated based on your dataset size (see Step 4.1)  
- We recommend at least **3 epochs** for training  

---

### 4.4: Submit the Job

Submit your SLURM job:

```bash
sbatch train_job.slurm
```

**Output:**

Training will generate:

- **Checkpoint file:** `<output_dir>/<run_id>/state_step<STEP>.pt`  
- **Log file:** `<output_dir>/<run_id>.txt`  

The checkpoint file contains the trained model weights needed for fine-tuning (Step 5).

## Step 5: Fine-tuning

Fine-tune the pretrained model on a classification task (e.g., XNLI) using a single GPU via SLURM.

---

### 5.1: Create Fine-tuning SLURM Script

Create a SLURM batch script (e.g., `finetune_job.slurm`) to launch fine-tuning:

```bash
#!/bin/bash
#SBATCH --job-name=finetune-batched
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --time=15:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "Start: $(date)"
echo "Job $SLURM_JOB_ID on $SLURM_JOB_NODELIST"
cd "/path/to/modded-nanogpt"

# Activate conda environment
module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ipagpt_training

# Unbuffered output for real-time logging
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

echo "Starting fine-tuning..."

python -u finetune.py   --no_subset   --dataset xnli-en-es-normal   --parent_dataset mugezhang/xnli-en-es-ipa   --pretrained_ckpt_path /path/to/pretrained/checkpoint.pt   --out_dir /path/to/finetuned/output   --tokenizer_dir /path/to/tokenizers   --tokenizer_name bpe-eng-spa-tokenizer.json   --vocab_size 50000   --hf_cache /path/to/cache   --data_dir /path/to/finetune/datasets   --text_column premise hypothesis   --label_column label   --num_classes 3   --batch_size 16   --num_epochs 3   --eval_interval 100   --eval_iters 10

echo "End: $(date)"
```

---

### 5.2: Key Fine-tuning Arguments

| Argument | Required | Description | Example |
|-----------|-----------|-------------|----------|
| `--dataset` | Yes | Dataset name for saving checkpoints | `xnli-en-es-normal` |
| `--parent_dataset` | Yes | HuggingFace dataset identifier | `mugezhang/xnli-en-es-ipa` |
| `--pretrained_ckpt_path` | Yes | Path to pretrained checkpoint from Step 4 | `/path/to/state_step050083.pt` |
| `--tokenizer_dir` | Yes | Directory containing tokenizer file | `/path/to/tokenizers` |
| `--tokenizer_name` | Yes | Tokenizer filename | `bpe-eng-spa-tokenizer.json` |
| `--vocab_size` | Yes | Must match pretrained model vocab size | `50000` |
| `--text_column` | Yes | Column(s) containing text data (space-separated for multiple) | `premise hypothesis` |
| `--label_column` | No | Column containing labels (default: label) | `label` |
| `--num_classes` | No | Number of classification classes (default: 2) | `3` |
| `--batch_size` | No | Training batch size (default: 16) | `16` |
| `--num_epochs` | No | Number of training epochs (default: 1) | `3` |
| `--eval_interval` | No | Evaluate every N iterations (default: 100) | `100` |
| `--eval_iters` | No | Number of iterations for evaluation (default: 10) | `10` |
| `--out_dir` | No | Output directory for checkpoints (default: ./checkpoints) | `/path/to/output` |
| `--data_dir` | No | Directory for processed datasets (default: ./datasets) | `/path/to/datasets` |
| `--hf_cache` | No | HuggingFace cache directory (default: ./cache) | `/path/to/cache` |
| `--no_subset` | No | Flag: dataset has no subset configuration | *(no value needed)* |

**Important Notes:**

- `--vocab_size` must match the pretrained model (from Step 4)  
- `--text_column` can accept multiple columns (e.g., `premise hypothesis` for XNLI)  
- Use `--no_subset` when your HuggingFace dataset doesn't have configuration subsets  
- The pretrained checkpoint path should point to the final checkpoint from Step 4 (e.g., `state_step050083.pt`)  

---

### 5.3: Submit the Job

Submit your fine-tuning job:

```bash
sbatch finetune_job.slurm
```

**Output:**

Fine-tuning will generate:

- **Best checkpoint:** `<out_dir>/<dataset>-ckpt.pt`  
- **Training logs:** in the SLURM output file  

The checkpoint file contains:

- Fine-tuned model weights  
- Validation metrics (loss, accuracy, F2 score)  
- Training configuration  

This checkpoint will be used for evaluation in **Step 6**.

## Step 6: Evaluation

Evaluate the fine-tuned model on test datasets to measure classification performance.

---

### Usage

Run the evaluation script with your fine-tuned checkpoint:

```bash
python evaluate.py   --checkpoint_path /path/to/finetuned/checkpoint.pt   --datasets <DATASET1> <DATASET2> ...   --dataset_labels <LABEL1> <LABEL2> ...   --batch_size 32   --output_dir ./evaluation_results
```

**Example:**

```bash
python evaluate.py   --checkpoint_path /fs/scratch/PAS2836/mugezhang/ipa_gpt_models/fine_tuned_cpt/normal_eng_spa/xnli-en-es-normal-ckpt.pt   --datasets iggy12345/xnli-en-ipa iggy12345/xnli-es-ipa   --dataset_labels English Spanish   --batch_size 32   --output_dir ./evaluation_results
```

---

### Key Arguments

| Argument | Required | Description | Default |
|-----------|-----------|-------------|----------|
| `--checkpoint_path` | Yes | Path to fine-tuned checkpoint from Step 5 | N/A |
| `--datasets` | Yes | HuggingFace dataset name(s) to evaluate (space-separated for multiple) | N/A |
| `--dataset_labels` | Yes | Friendly labels for each dataset (must match number of datasets) | N/A |
| `--dataset_config` | No | Dataset configuration/subset (for datasets with multiple configs) | None |
| `--split` | No | Dataset split to evaluate on | test |
| `--batch_size` | No | Batch size for evaluation | 32 |
| `--device` | No | Device to run evaluation on | cuda |
| `--dtype` | No | Data type for evaluation | bfloat16 |
| `--output_dir` | No | Directory to save detailed results (JSON) | None |

---

**Important Notes:**

- The number of `--dataset_labels` must match the number of `--datasets`  
- The script will evaluate on all specified datasets sequentially  
- Evaluation uses the **test split** by default (change with `--split`)  

---

### Output

The evaluation script will output results for each dataset:

#### Console Output:

- Overall metrics (Accuracy, F1, F2, Recall)  
- Classification report (per-class precision, recall, F1)  
- Confusion matrix  
- Per-class accuracy  

#### JSON Output (if `--output_dir` specified):

Detailed results are saved as JSON files containing:

- Checkpoint path and configuration  
- Predictions and labels for each example  
- All computed metrics  
- Confusion matrices  

Files are named based on the checkpoint type (e.g., `results_normal.json` or `results_phonemized.json`).
