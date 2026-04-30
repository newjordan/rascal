"""Compare compile modes and kernel variants on v5 model."""
import os, sys, time, importlib.util, gc
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

B, T = h.train_batch_tokens // h.grad_accum_steps // h.train_seq_len, h.train_seq_len
input_ids = torch.randint(0, h.vocab_size, (B, T), device=device)
target_ids = torch.randint(0, h.vocab_size, (B, T), device=device)

def benchmark(label, build_compiled, n=30):
    gc.collect(); torch.cuda.empty_cache()
    torch.manual_seed(42)
    base = m.GPT(h).to(device).bfloat16()
    m.restore_fp32_params(base)
    compiled = build_compiled(base)
    try:
        for _ in range(5):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = compiled(input_ids, target_ids)
            loss.backward()
            base.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = compiled(input_ids, target_ids)
            loss.backward()
            base.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / n
        tps = B*T/elapsed
        full_step = elapsed * 8
        print(f'{label:35s}: {elapsed*1000:6.1f}ms/microbatch, {tps:>10,.0f} tok/s, full step {full_step*1000:.0f}ms')
    except Exception as e:
        print(f'{label:35s}: FAILED ({type(e).__name__}: {str(e)[:80]})')
    finally:
        del base, compiled
        gc.collect(); torch.cuda.empty_cache()

print('=== Compile mode comparison ===')
print(f'B={B}, T={T}, total tokens/microbatch={B*T}')

benchmark('default',
    lambda b: torch.compile(b, dynamic=False, fullgraph=True))

benchmark('mode=max-autotune-no-cudagraphs',
    lambda b: torch.compile(b, dynamic=False, fullgraph=True, mode='max-autotune-no-cudagraphs'))

# Skip reduce-overhead — known to break with cached Rotary tensors
print('(skipping reduce-overhead and max-autotune — break with cached Rotary tensors)')
