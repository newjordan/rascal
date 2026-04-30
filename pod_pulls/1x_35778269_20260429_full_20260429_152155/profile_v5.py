"""Profile a single training step on v5 model. Find actual GPU hot spots."""
import os, sys, time, importlib.util
os.environ.setdefault('DATA_DIR', '/workspace/Fartmagic/data/')
os.environ.setdefault('NUM_LOOPS', '0')
os.environ.setdefault('SLIDING_WINDOW_ENABLED', '0')
os.environ.setdefault('EMA_DECAY', '0')

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

B, T = h.train_batch_tokens // h.grad_accum_steps // h.train_seq_len, h.train_seq_len
print(f'micro batch shape: ({B}, {T})  seqlen={T}  total tokens/microbatch={B*T}')
input_ids = torch.randint(0, h.vocab_size, (B, T), device=device)
target_ids = torch.randint(0, h.vocab_size, (B, T), device=device)

# warmup compile
for _ in range(3):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = compiled(input_ids, target_ids)
    loss.backward()
    base.zero_grad(set_to_none=True)
torch.cuda.synchronize()

# time microbatch (forward + backward)
N = 30
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(N):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = compiled(input_ids, target_ids)
    loss.backward()
    base.zero_grad(set_to_none=True)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - t0) / N
print(f'{N} microbatch fwd+bwd avg: {elapsed*1000:.1f}ms')
print(f'tok/s/microbatch: {B*T/elapsed:,.0f}')
print(f'expected step time (8 microbatches): {elapsed*8*1000:.1f}ms')
print(f'expected step throughput (786432 tokens): {786432/(elapsed*8):,.0f} tok/s')

# Profile to find hot kernels
from torch.profiler import profile, record_function, ProfilerActivity
print('\n=== Profile (forward + backward) ===')
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, with_stack=False) as prof:
    for _ in range(5):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = compiled(input_ids, target_ids)
        loss.backward()
        base.zero_grad(set_to_none=True)
torch.cuda.synchronize()
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=25))
