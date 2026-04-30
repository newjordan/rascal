"""Bench QKV packing impact on fwd+bwd time."""
import os, sys, time, importlib.util, gc
os.environ['DATA_DIR'] = '/workspace/Fartmagic/data/'
os.environ['NUM_LOOPS'] = '0'

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
torch._dynamo.config.optimize_ddp = False

device = torch.device('cuda', 0); torch.cuda.set_device(device)

def bench(label, pack, n=30):
    os.environ['PACK_QKV'] = '1' if pack else '0'
    # reload module
    if 'm' in globals(): del globals()['m']
    spec = importlib.util.spec_from_file_location('m', '/workspace/Mikey_II_v5/train_gpt.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    h = m.Hyperparameters()
    torch.manual_seed(42)
    base = m.GPT(h).to(device).bfloat16()
    m.restore_fp32_params(base)
    compiled = torch.compile(base, dynamic=False, fullgraph=True)
    B, T = 48, 2048
    input_ids = torch.randint(0, h.vocab_size, (B, T), device=device)
    target_ids = torch.randint(0, h.vocab_size, (B, T), device=device)
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
    print(f'{label}: {elapsed*1000:.1f}ms/microbatch, {B*T/elapsed:,.0f} tok/s')
    del base, compiled
    gc.collect(); torch.cuda.empty_cache()
    return elapsed

t_default = bench('default (separate q,k,v)', pack=False)
t_pack = bench('PACK_QKV=1', pack=True)
speedup = (t_default - t_pack) / t_default * 100
print(f'Speedup: {speedup:+.2f}%')
