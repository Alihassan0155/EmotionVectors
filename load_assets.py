import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

sst2 = load_dataset("stanfordnlp/sst2")
tweet_eval = load_dataset("cardiffnlp/tweet_eval", "sentiment")

print("SST-2:", {split: len(rows) for split, rows in sst2.items()})
print("TweetEval sentiment:", {split: len(rows) for split, rows in tweet_eval.items()})

models = [
    ("Qwen/Qwen2.5-0.5B-Instruct", AutoModelForCausalLM),
    ("Qwen/Qwen2.5-1.5B-Instruct", AutoModelForCausalLM),
    ("google-bert/bert-large-uncased", AutoModelForMaskedLM),
]

device = "cuda" if torch.cuda.is_available() else "cpu"

for model_id, model_class in models:
    print(f"Loading {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = model_class.from_pretrained(
        model_id,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    print(f"Loaded {model_id}")

    # Release this model before loading the next one.
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
