# Checkpoint Format Comparison

## Quick Reference

| Feature | train_gpt_small.py | train_gpt_medium.py | GPTBatched |
|---------|-------------------|---------------------|------------|
| Attention weights | `qkv_w` [3,hdim,dim]<br>+ `c_proj.weight` [dim,hdim] | `qkvo_w` [4,hdim,dim] | `qkvo_w` [4,hdim,dim] |
| MLP weights | `c_fc.weight` [hdim,dim]<br>+ `c_proj.weight` [dim,hdim] | `fc_w` [hdim,dim]<br>+ `proj_w` [dim,hdim] | `fc_w` [hdim,dim]<br>+ `proj_w` [dim,hdim] |
| Conversion needed? | ✓ Yes | ✗ No | N/A |

## What This Means

### Small Model Checkpoints
- Use **legacy format** with separate Q/K/V and output projection
- Use **layer wrappers** (CastedLinear) for projections
- **Require conversion** to work with GPTBatched
- Conversion handled automatically by `load_checkpoint_helper.py`

### Medium Model Checkpoints
- Use **modern format** matching GPTBatched directly
- Use **direct parameters** without layer wrappers
- **Load directly** into GPTBatched - no conversion needed!
- More efficient loading

## Loading Either Format

The `load_checkpoint_helper.py` module automatically:
1. Detects which format the checkpoint uses
2. Loads small model checkpoints via GPTFlexAttention → GPTBatched conversion
3. Loads medium model checkpoints directly into GPTBatched
4. Handles vocab_size padding and scalars mismatches

## Example Usage

```python
from load_checkpoint_helper import load_pretrained_checkpoint

# Works for BOTH small and medium model checkpoints!
model, info, checkpoint = load_pretrained_checkpoint(
    '/path/to/checkpoint.pt',
    device='cuda'
)

# Output shows which format was detected:
# Format: small (from train_gpt_small.py)
# or
# Format: medium (from train_gpt_medium.py)
```

## Checkpoint Locations

Your checkpoints should be under:
```
/fs/scratch/PAS2836/mugezhang/clean_bbpe_50k/
├── eng_spa_owt_run_*/best_state_*.pt       (small or medium?)
├── hin_urd_owt_run_*/best_state_*.pt       (small or medium?)
├── rus_pol_owt_run_*/best_state_*.pt       (small or medium?)
└── tam_mal_owt_run_*/best_state_*.pt       (small or medium?)
```

To check which format a checkpoint uses:
```bash
python test_checkpoint_loading.py /path/to/checkpoint.pt
```

The output will show:
```
Format: small (from train_gpt_small.py)
```
or
```
Format: medium (from train_gpt_medium.py)
```
