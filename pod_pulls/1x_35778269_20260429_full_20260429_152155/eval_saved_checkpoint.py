"""
Empirical test: load Raphe II seed42 final_model.pt and compute val_loss + val_bpb
on the local sp4096 val data using BOTH tokenizers' LUTs.

If val_loss matches published ~2.51 → model is compatible with these (ffeeea3) tokens
                                       → records' shards were ffeeea3-tokenized
                                       → mismatch confirmed (records' bpb is computed
                                          with 51cca47 LUT against ffeeea3 shards)
If val_loss is much higher → model trained on different vocab → records used different shards
"""
import os, sys, math, glob, hashlib
import numpy as np, torch
import sentencepiece as spm
from pathlib import Path

sys.path.insert(0, '/workspace/raphe_ii')
import importlib.util
spec = importlib.util.spec_from_file_location("train_gpt", "/workspace/raphe_ii/train_gpt.py")

# Patch out the distributed-launcher main() at import — we just want the classes
import builtins
_orig_name = '__main__'
mod = importlib.util.module_from_spec(spec)
mod.__name__ = "train_gpt_lib"
# Hack: prevent if __name__=='__main__' main() from firing
src = Path("/workspace/raphe_ii/train_gpt.py").read_text()
src = src.replace("if __name__=='__main__':main()", "")
exec(compile(src, "/workspace/raphe_ii/train_gpt.py", "exec"), mod.__dict__)

H = mod.Hyperparameters

# Override h for single-GPU eval
class h:
    pass
for k,v in vars(H).items():
    if not k.startswith("_"):
        setattr(h, k, v)
h.distributed = False
h.rank = 0
h.world_size = 1
h.local_rank = 0
h.is_main_process = True
h.grad_accum_steps = 1
h.data_dir = "/workspace/data/"
h.datasets_dir = os.path.join(h.data_dir, "datasets", f"fineweb10B_sp{h.vocab_size}")
h.train_files = os.path.join(h.datasets_dir, "fineweb_train_*.bin")
h.val_files = os.path.join(h.datasets_dir, "fineweb_val_*.bin")
h.tokenizer_path = os.path.join(h.data_dir, "tokenizers", f"fineweb_{h.vocab_size}_bpe.model")

device = torch.device("cuda")
torch.cuda.set_device(0)

print(f"vocab_size = {h.vocab_size}")
print(f"tokenizer = {h.tokenizer_path}  md5={hashlib.md5(open(h.tokenizer_path,'rb').read()).hexdigest()[:8]}")
print(f"val files = {h.val_files}")

val_data = mod.ValidationData(h, device)
print(f"val tokens: {val_data.val_tokens.numel()-1}")

# Build model
print("Constructing model...")
base_model = mod.GPT(h).to(device)

# Load checkpoint
print("Loading checkpoint /workspace/raphe_ii_seed42_final_model.pt ...")
state = torch.load("/workspace/raphe_ii_seed42_final_model.pt", map_location="cpu", weights_only=False)
if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]
elif isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
missing, unexpected = base_model.load_state_dict(state, strict=False)
print(f"  missing keys: {len(missing)}  unexpected: {len(unexpected)}")
if missing[:3]: print(f"  missing[:3]: {missing[:3]}")
if unexpected[:3]: print(f"  unexpected[:3]: {unexpected[:3]}")
base_model.eval()

# Run val under current tokenizer LUT (51cca47 — what records used)
print("\n=== Running eval_val with currently-loaded tokenizer LUT ===")
val_loss, val_bpb = mod.eval_val(h, device, val_data, base_model)
print(f"  val_loss: {val_loss:.6f}")
print(f"  val_bpb:  {val_bpb:.6f}")
print(f"  published seed42 pre-quant post-ema: val_loss=2.51231130 val_bpb=0.85660134")
print(f"  → val_loss diff vs published: {val_loss - 2.51231130:+.4f}")
