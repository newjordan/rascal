# Rascal SP4096 Loop 1x Smoke

Run class: `mechanics_proxy_1x_smoke`.

Source body: `quarantine/2026-04-28_tokenizer_provenance/candidate_submissions/Raphe_1.1/train_gpt_8xgpu.py` (`b93227b9f394b81fa77c412e0ecc394f17fb9effb250b96fe77903b9e1687d39`). This is the SP4096 Raphe/Rascal-style body, not Mikey.

Evidence gate: `notes/2026-04-29_kernel_loop_mining_brief.md` (`e3047ae3c94455be41c601fb023cc5e855a94e4e54c6fe618a6606efbae64689`). Per that gate, this is FA3/control loop topology only. No custom kernel integration is prepared here, and `MLP_KERNEL_MODE` fails closed if set.

Loop topology: Donnie/Rascal-style `NUM_LOOPS=2`, `LOOP_START=2`, `LOOP_END=4`, `NUM_LAYERS=11`, with 1x smoke activation at `ENABLE_LOOPING_AT=0.0` so the loop path is exercised during the short run.

Command:

```bash
cd /home/frosty40/sota_rascal/legs/2026-04-29_rascal_sp4096_loop_1x_smoke
./launch_gpu0_smoke.sh
```
