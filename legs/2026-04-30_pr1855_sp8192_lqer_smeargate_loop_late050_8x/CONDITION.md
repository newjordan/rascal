# PR1855 SP8192 CaseOps Loop Late050

Run label: `new_experiment`, `standard_8x`.

Parent: `legs/2026-04-30_pr1855_sp8192_lqer_smeargate_repro_8x/run.py`

Parent SHA256: `454f710d174be80f4603069ca952833d694f60d1d34c0c25703528323bc8878b`

## Hypothesis

Keep the accepted PR1855 CaseOps/LQER/SparseGate/SmearGate stack and test only
the loop timing axis that showed useful behavior in the local Mikey work.

Changed field:

- `enable_looping_at: 0.35 -> 0.50`

Unchanged:

- SP8192 CaseOps tokenizer/data
- `num_loops=2`, `loop_start=3`, `loop_end=5`
- loop warmup remains enabled
- pergroup compression
- LQER asym
- sparse attention gate
- SmearGate
- phased LoRA TTT
- seed `42`

## Command

```bash
cd /workspace/sota_rascal/legs/2026-04-30_pr1855_sp8192_lqer_smeargate_loop_late050_8x
./launch_8x.sh
tail -f logs/pr1855_sp8192_lqer_smeargate_loop_late050_8x_seed42.txt
```
