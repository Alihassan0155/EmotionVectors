"""Evaluate both Qwen models with one fixed prompt and greedy decoding."""

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import torch

from load_assets import load_model, load_sentiment_datasets
from prompt_templates import PROMPT_TEMPLATES, render_prompt


TASKS = {
    "sst2": ("validation", "sentence"),
    "tweet_eval_sentiment": ("test", "text"),
}


def predict_label(model, tokenizer, dataset_name, template_id, text, labels):
    prompt = render_prompt(dataset_name, template_id, text)
    messages = [{"role": "user", "content": prompt}]
    chat = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,  # Greedy decoding: equivalent to temperature=0.
            max_new_tokens=4,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(
        output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip().lower()
    answer = answer.strip(" .,!?:;\"'`\n\t")
    return answer if answer in labels else "INVALID"


def metrics(gold, predicted, labels):
    total = len(gold)
    correct = sum(a == b for a, b in zip(gold, predicted))
    per_label = {}
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(gold, predicted))
        fp = sum(a != label and b == label for a, b in zip(gold, predicted))
        fn = sum(a == label and b != label for a, b in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(a == label for a in gold)}
    return {
        "examples": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(item["f1"] for item in per_label.values()) / len(labels),
        "invalid_predictions": Counter(predicted)["INVALID"],
        "per_label": per_label,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-id", choices=tuple(PROMPT_TEMPLATES["sst2"]), default="template_1")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N rows of each split")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    datasets = load_sentiment_datasets()
    output_dir = ROOT / "artifacts"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / f"predictions_{args.template_id}.csv"
    results = {"template_id": args.template_id, "decoding": "greedy", "max_new_tokens": 4, "results": {}}

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "model", "dataset", "split", "row_index", "gold_label", "predicted_label"
        ])
        writer.writeheader()
        for model_size in ("0.5b", "1.5b"):
            model, tokenizer = load_model(model_size, args.device)
            model_results = {}
            for dataset_name, (split, text_column) in TASKS.items():
                rows = datasets[dataset_name][split]
                labels = datasets[dataset_name]["train"].features["label"].names
                count = min(len(rows), args.limit) if args.limit else len(rows)
                gold, predicted = [], []
                for row_index in range(count):
                    row = rows[row_index]
                    if row["label"] < 0:
                        continue
                    gold_label = labels[row["label"]]
                    prediction = predict_label(
                        model, tokenizer, dataset_name, args.template_id,
                        row[text_column], labels,
                    )
                    gold.append(gold_label)
                    predicted.append(prediction)
                    writer.writerow({
                        "model": model_size, "dataset": dataset_name, "split": split,
                        "row_index": row_index, "gold_label": gold_label,
                        "predicted_label": prediction,
                    })
                file.flush()
                model_results[dataset_name] = metrics(gold, predicted, labels)
                result = model_results[dataset_name]
                print(f"{model_size} {dataset_name}: accuracy={result['accuracy']:.3f}, macro_f1={result['macro_f1']:.3f}, invalid={result['invalid_predictions']}", flush=True)
            results["results"][model_size] = model_results
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metrics_path = output_dir / f"metrics_{args.template_id}.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved {csv_path} and {metrics_path}")


if __name__ == "__main__":
    main()
