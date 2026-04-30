"""Profile grad_accum=1 case (matches 8x per-GPU pattern). Find kernel-level bottlenecks."""
import os, sys, time, importlib.util
os.environ.setdefault('DATA_DIR', '/workspace/Fartmagic/data/')
os.environ.setdefault('NUM_LOOPS', '0')

import torch

spec = importlib.util.spec_from_file_location('m', '/workspace/Mikey_II_v5/train_gpt.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

h = m.Hyperparameters()
device = torch.device('cuda', 0); torch.cuda.set_device(device)
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
torch._dynamo.config.optimize_ddp = False

base = m.GPT(h).to(device).bfloat16()
m.restore_fp32_params(base)
compiled = torch.compile(base, dynamic=False, fullgraph=True)

# Match 8x microbatch: 786432/8 = 98304 tokens = 48 sequences x 2048
B, T = 48, 2048
input_ids = torch.randint(0, h.vocab_size, (B, T), device=device)
target_ids = torch.randint(0, h.vocab_size, (B, T), device=device)

print(f'B={B} T={T} (matches 8x microbatch)')

for _ in range(5):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = compiled(input_ids, target_ids)
    loss.backward()
    base.zero_grad(set_to_none=True)
torch.cuda.synchronize()

N = 30
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(N):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = compiled(input_ids, target_ids)
    loss.backward()
    base.zero_grad(set_to_none=True)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - t0) / N
print(f'fwd+bwd (no opt): {elapsed*1000:.1f}ms, {B*T/elapsed:,.0f} tok/s/microbatch')

from torch.profiler import profile, ProfilerActivity
print('\n=== Profile ===')
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
    for _ in range(5):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = compiled(input_ids, target_ids)
        loss.backward()
        base.zero_grad(set_to_none=True)
torch.cuda.synchronize()
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=20))
