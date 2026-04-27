# Raphe

| seed | val_bpb (sliding) | bytes |
|------|-------------------|-------|
| 42   | pending           | pending |
| 300  | pending           | pending |
| 444  | 0.87168317        | 13,487,656 |

```
torchrun --standalone --nproc_per_node=8 train_gpt_8xgpu.py
```
