#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/workspace/SOTA_FINAL/data"
CASEOPS_ROOT="${DATA_ROOT}/datasets/fineweb10B_sp10240_caseops/datasets"
DATASET_DIR="${CASEOPS_ROOT}/datasets/fineweb10B_sp10240_lossless_caps_caseops_v1_reserved"
TOKENIZER_PATH="${CASEOPS_ROOT}/tokenizers/fineweb_10240_bpe_lossless_caps_caseops_v1_reserved.model"

export PIP_ROOT_USER_ACTION=ignore
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "[1/5] system packages"
DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq lrzip

echo "[2/5] /venv/main"
if [[ ! -x /venv/main/bin/python3 ]]; then
  python3 -m venv /venv/main
fi
/venv/main/bin/python3 -m pip install --upgrade pip -q

if ! /venv/main/bin/python3 - <<'PY' 2>/dev/null
import torch
assert torch.__version__.endswith("+cu130"), torch.__version__
assert torch.version.cuda == "13.0", torch.version.cuda
assert torch.cuda.device_count() >= 6, torch.cuda.device_count()
PY
then
  echo "installing torch cu130"
  /venv/main/bin/python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 torch
fi

echo "[3/5] python packages + FA3"
/venv/main/bin/python3 -m pip install --no-cache-dir \
  numpy sentencepiece brotli huggingface_hub hf_transfer tqdm python-minifier zstandard
if ! /venv/main/bin/python3 - <<'PY' 2>/dev/null
import flash_attn_interface
PY
then
  /venv/main/bin/python3 -m pip install --no-cache-dir \
    https://download.pytorch.org/whl/cu130/flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl
fi

echo "[4/5] CaseOps dataset"
mkdir -p "${CASEOPS_ROOT}"
/venv/main/bin/python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Frosty40/10k_caseops_golfer",
    repo_type="dataset",
    local_dir="/workspace/SOTA_FINAL/data/datasets/fineweb10B_sp10240_caseops/datasets",
)
PY

echo "[5/5] verify"
/venv/main/bin/python3 - <<PY
from pathlib import Path
import sentencepiece as spm
import torch

dataset_dir = Path("${DATASET_DIR}")
tokenizer_path = Path("${TOKENIZER_PATH}")
train = sorted(dataset_dir.glob("fineweb_train_*.bin"))
val = sorted(p for p in dataset_dir.glob("fineweb_val_*.bin") if "val_bytes" not in p.name)
val_bytes = sorted(dataset_dir.glob("fineweb_val_bytes_*.bin"))
sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))

print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
print("vocab", sp.vocab_size(), "caseops", [sp.piece_to_id(chr(0xE001 + i)) for i in range(4)])
print("train", len(train), "val", len(val), "val_bytes", len(val_bytes))

assert torch.__version__.endswith("+cu130"), torch.__version__
assert torch.version.cuda == "13.0", torch.version.cuda
assert torch.cuda.device_count() >= 6, torch.cuda.device_count()
assert sp.vocab_size() == 10240, sp.vocab_size()
assert [sp.piece_to_id(chr(0xE001 + i)) for i in range(4)] == [4, 5, 6, 7]
assert len(train) >= 80, len(train)
assert len(val) >= 1, len(val)
assert len(val_bytes) == len(val), (len(val_bytes), len(val))
PY

echo "READY"
