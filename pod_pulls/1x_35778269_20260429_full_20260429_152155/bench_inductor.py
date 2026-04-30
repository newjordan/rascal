"""Bench inductor flags impact."""
import os, sys, time, importlib.util, gc
os.environ['DATA_DIR'] = '/workspace/Fartmagic/data/'
os.environ['NUM_LOOPS'] = '0'
os.environ['PACK_QKV'] = '0'

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
torch._dynamo.config.optimize_ddp = False

device = torch.device('cuda', 0); torch.cuda.set_device(device)

def bench(label, pre_compile_setup, n=30):
    pre_compile_setup()
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
    torch._dynamo.reset()
    gc.collect(); torch.cuda.empty_cache()
    return elapsed

def baseline():
    pass  # default

def coord_descent():
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.coordinate_descent_check_all_directions = True

def epilogue_first():
    torch._inductor.config.epilogue_fusion_first = True

def reset_flags():
    torch._inductor.config.coordinate_descent_tuning = False
    torch._inductor.config.coordinate_descent_check_all_directions = False
    torch._inductor.config.epilogue_fusion_first = False

t_baseline = bench('baseline', baseline)
reset_flags()

t_coord = bench('coordinate_descent_tuning', coord_descent)
print(f'coord_descent: {(t_baseline-t_coord)/t_baseline*100:+.2f}%')
reset_flags()

t_epi = bench('epilogue_fusion_first', epilogue_first)
print(f'epilogue_first: {(t_baseline-t_epi)/t_baseline*100:+.2f}%')
reset_flags()

def both():
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.epilogue_fusion_first = True

t_both = bench('coord+epi', both)
print(f'coord+epi: {(t_baseline-t_both)/t_baseline*100:+.2f}%')
