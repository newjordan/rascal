"""
Generate text from the saved Raphe II seed42 checkpoint using 51cca47 tokenizer.
If output is coherent English -> model was trained on 51cca47-tokenized data.
If output is gibberish -> model + tokenizer are mismatched.
"""
import os, sys, hashlib
import torch
import sentencepiece as spm
from pathlib import Path

# Load the matching train_gpt.py (mlp_mult=4.5 safemini variant — matches checkpoint)
import importlib.util
script_path = "/workspace/raphe_ii/train_gpt.py"
src = Path(script_path).read_text().replace("if __name__=='__main__':main()", "")
mod = type(sys)('train_gpt_lib')
mod.__file__ = script_path
exec(compile(src, script_path, "exec"), mod.__dict__)

H = mod.Hyperparameters
class h: pass
for k,v in vars(H).items():
    if not k.startswith("_"): setattr(h, k, v)
h.distributed = False; h.rank = 0; h.world_size = 1; h.local_rank = 0
h.is_main_process = True; h.grad_accum_steps = 1
h.tokenizer_path = "/workspace/data/tokenizers/fineweb_4096_bpe.model"

device = torch.device("cuda")
torch.cuda.set_device(0)

tok_md5 = hashlib.md5(open(h.tokenizer_path,"rb").read()).hexdigest()
print(f"tokenizer: {h.tokenizer_path}")
print(f"  md5: {tok_md5}")

sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
sp_alt = spm.SentencePieceProcessor(model_file="/tmp/sp4096_ffeeea3.model")
print(f"  vocab_size: {sp.vocab_size()}")
print(f"  alt tokenizer (ffeeea3): {hashlib.md5(open('/tmp/sp4096_ffeeea3.model','rb').read()).hexdigest()}")
print()

print("Building model...")
model = mod.GPT(h).to(device)
print("Loading checkpoint...")
state = torch.load("/workspace/raphe_ii_seed42_final_model.pt", map_location="cpu", weights_only=False)
if isinstance(state, dict) and "model_state_dict" in state: state = state["model_state_dict"]
elif isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
model.load_state_dict(state, strict=True)
model.eval()
print("Loaded.\n")

PROMPTS = [
    "The Roman Empire was",
    "Scientists have discovered",
    "Once upon a time, in a small village",
    "The recipe for chocolate cake calls for",
]

@torch.no_grad()
def generate(prompt, max_new_tokens=60, temperature=0.7, encode_with='51cca47'):
    encoder = sp if encode_with == '51cca47' else sp_alt
    ids = [1] + encoder.encode(prompt)  # BOS + encoded prompt
    print(f"  prompt: {prompt!r}  (encoded with {encode_with})")
    print(f"  encoded: {ids[:20]}{'...' if len(ids) > 20 else ''}")
    x = torch.tensor([ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            if hasattr(model, 'forward_logits'):
                logits = model.forward_logits(x)
            else:
                logits = model(x, x)
        next_logits = logits[0, -1].float() / temperature
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        ids.append(next_id)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        if next_id == 2: break  # EOS
    text_51 = sp.decode(ids)
    text_ff = sp_alt.decode(ids)
    print(f"  decoded with 51cca47: {text_51!r}")
    print(f"  decoded with ffeeea3: {text_ff!r}")
    print()

print("=== ENCODE PROMPT WITH 51cca47 ===\n")
for p in PROMPTS[:2]:
    generate(p, encode_with='51cca47', temperature=0.4)

print("\n=== ENCODE PROMPT WITH ffeeea3 ===\n")
for p in PROMPTS[:2]:
    generate(p, encode_with='ffeeea3', temperature=0.4)
