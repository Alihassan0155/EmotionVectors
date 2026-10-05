"""Check NNsight residual extraction against HF hidden states for one text."""

import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(r"D:\EmotionVectors") / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from nnsight import NNsight
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
inputs = tok("I was so happy to see my family.", add_special_tokens=False,
             return_tensors="pt").to("cuda")
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                             local_files_only=True).eval().to("cuda")
wrapped = NNsight(base)
mask = inputs["attention_mask"].unsqueeze(-1)
with torch.inference_mode():
    with wrapped.trace(inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False):
        pooled_layers = []
        for layer_index in range(len(base.model.layers)):
            residual = wrapped.model.layers[layer_index].output
            pooled_layers.append((residual.float() * mask).sum(1) / mask.sum(1))
        all_pooled = torch.stack(pooled_layers, dim=1).cpu().save()
    with wrapped.trace(inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False):
        first_residual = wrapped.model.layers[0].output
        first = first_residual.save()
        first_pooled = ((first_residual.float() * mask).sum(1)
                        / mask.sum(1)).save()
        last = wrapped.model.layers[-1].output.save()
    direct = base(**inputs, use_cache=False, output_hidden_states=True)
print("first:", first.shape, "last:", last.shape)
print("all pooled:", all_pooled.shape)
print("first max difference:", (first - direct.hidden_states[1]).abs().max().item())
print("pooled max difference:",
      (first_pooled - (first.float() * mask).sum(1) / mask.sum(1)).abs().max().item())
print("last after norm max difference:",
      (base.model.norm(last) - direct.hidden_states[-1]).abs().max().item())
