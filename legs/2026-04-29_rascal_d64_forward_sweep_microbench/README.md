# Rascal D64 Forward Sweep Microbench

Purpose: tune the whale forward side for the exact active Rascal 4k shape before any training integration.

- Run label: `mechanics_1x_forward_kernel_scout`
- Source body: `quarantine/2026-04-28_tokenizer_provenance/candidate_submissions/Raphe_1.1/train_gpt_8xgpu.py`
- Source hash: `b93227b9f394b81fa77c412e0ecc394f17fb9effb250b96fe77903b9e1687d39`
- Shape: `B=48,T=2048,H=8,KV=4,D=64`
- Tokenizer/data provenance: SP4096 paths are recorded, but this is a tensor-only kernel microbench.
- Kernel path: `vault/whale_kernel_triton.py`
- Comparator: FA3 forward and full FA3 forward+backward at the same tensor shape.

Promotion rule: do not integrate into a training leg unless the full `whale_fa3_bwd` row beats FA3 median on this shape. Forward-only wins are evidence for more tuning, not enough by themselves.

Run on pod GPU0:

```bash
cd /workspace/sota_rascal/legs/2026-04-29_rascal_d64_forward_sweep_microbench
./launch_gpu0_forward_sweep.sh
```
