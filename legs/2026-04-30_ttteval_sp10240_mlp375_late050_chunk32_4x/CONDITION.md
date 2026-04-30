# TTT Eval Chunk32 - SP10240 CaseOps MLP3.75 Late050

Run label: `mechanics_4x_ttt_eval`.

This is an eval-only mechanics test. It reuses the under-cap quantized artifact
from `2026-04-30_pr1855_sp10240_caseops_mlp375_late050_8x` and changes only the
local phased-TTT chunk size from `48` to `32`.

## Fixed Condition

- Parent artifact: `/workspace/sota_rascal/legs/2026-04-30_pr1855_sp10240_caseops_mlp375_late050_8x/logs/final_model.int6.ptz`
- Dataset: `/workspace/SOTA_FINAL/data/datasets/fineweb10B_sp10240_caseops/datasets/datasets/fineweb10B_sp10240_lossless_caps_caseops_v1_reserved`
- Tokenizer: `/workspace/SOTA_FINAL/data/datasets/fineweb10B_sp10240_caseops/datasets/tokenizers/fineweb_10240_bpe_lossless_caps_caseops_v1_reserved.model`
- Vocab size: `10240`
- GPU count: `4`
- Eval seconds class: `1200` mechanics proxy
- Seed: `444`

## Changed Field

- `ttt_chunk_size=32`

## Command

```bash
cd /workspace/sota_rascal/legs/2026-04-30_ttteval_sp10240_mlp375_late050_chunk32_4x
./launch_4x_ttt_eval.sh
tail -f logs/ttteval_sp10240_mlp375_late050_chunk32_4x_seed444.txt
```
