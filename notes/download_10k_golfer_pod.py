from pathlib import Path
import os
import glob

from huggingface_hub import snapshot_download

base = Path("/workspace/SOTA_FINAL/data")
raw = base / "hf_10k_golfer"
tok = base / "tokenizers"
ds = base / "datasets" / "fineweb10B_sp10240"

raw.mkdir(parents=True, exist_ok=True)
tok.mkdir(parents=True, exist_ok=True)
ds.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="Frosty40/10k_golfer",
    repo_type="dataset",
    local_dir=str(raw),
    local_dir_use_symlinks=False,
    max_workers=8,
)

for name in ("fineweb_10240_bpe.model", "fineweb_10240_bpe.vocab"):
    target = tok / name
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(raw / name)

for path in glob.glob(str(raw / "fineweb_*.bin")):
    target = ds / os.path.basename(path)
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(Path(path))

train_count = len(list(ds.glob("fineweb_train_*.bin")))
val_count = len(list(ds.glob("fineweb_val_*.bin")))
print(f"download_and_links_done train={train_count} val={val_count}")
