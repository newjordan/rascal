"""Bench FP8 (torchao) vs bf16 fwd+bwd."""
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

def build_model():
    spec = importlib.util.spec_from_file_location('m', '/workspace/Mikey_II_v5/train_gpt.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    h = m.Hyperparameters()
    torch.manual_seed(42)
    base = m.GPT(h).to(device).bfloat16()
    m.restore_fp32_params(base)
    return m, h, base

def bench(label, base, n=30):
    compiled = torch.compile(base, dynamic=False, fullgraph=False)  # fp8 may need fullgraph=False
    B, T = 48, 2048
    spec = importlib.util.spec_from_file_location('m', '/workspace/Mikey_II_v5/train_gpt.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    h = m.Hyperparameters()
    input_ids = torch.randint(0, h.vocab_size, (B, T), device=device)
    target_ids = torch.randint(0, h.vocab_size, (B, T), device=device)
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
        print(f'{label}: {elapsed*1000:.1f}ms/microbatch, {B*T/elapsed:,.0f} tok/s, last loss {loss.item():.4f}')
        return elapsed
    except Exception as e:
        print(f'{label}: FAILED ({type(e).__name__}): {str(e)[:200]}')
        return None

# Baseline
m, h, base = build_model()
t_baseline = bench('bf16 baseline', base)
del base; gc.collect(); torch.cuda.empty_cache()

# FP8 (only MLP linears)
from torchao.float8 import convert_to_float8_training, Float8LinearConfig

def filter_only_mlp(mod, name):
    return '.mlp.' in name

m, h, base = build_model()
config = Float8LinearConfig.from_recipe_name('tensorwise')
convert_to_float8_training(base, config=config, module_filter_fn=filter_only_mlp)
print('Converted MLP linears to FP8.')
t_fp8_mlp = bench('FP8 (MLP only)', base)
del base; gc.collect(); torch.cuda.empty_cache()

if t_fp8_mlp and t_baseline:
    print(f'FP8 MLP speedup: {(t_baseline-t_fp8_mlp)/t_baseline*100:+.2f}%')

# All linears FP8
m, h, base = build_model()
convert_to_float8_training(base, config=config)  # all eligible linears
print('Converted all linears to FP8.')
t_fp8_all = bench('FP8 (all linears)', base)

if t_fp8_all and t_baseline:
    print(f'FP8 all speedup: {(t_baseline-t_fp8_all)/t_baseline*100:+.2f}%')
