"""Evaluate 12,000 stratified SST-2 training rows with all fixed prompts.

The 872-row validation split is excluded and remains a separate result.
Each prediction is committed to SQLite so the run can resume after interruption.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import torch
import transformers
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_templates import PROMPT_TEMPLATES, render_prompt

SAMPLE_SIZE = 12_000
SAMPLE_SEED = 42
TEMPLATE_IDS = tuple(PROMPT_TEMPLATES["sst2"])
MODEL_IDS = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}


def load_model(model_size, device):
    model_id = MODEL_IDS[model_size]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    ).eval()
    return model, tokenizer


def predict_label(model, tokenizer, template_id, text, labels):
    prompt = render_prompt("sst2", template_id, text)
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(chat, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs, do_sample=False, max_new_tokens=4,
            pad_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(
        output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip().lower().strip(" .,!?:;\"'`\n\t")
    return answer if answer in labels else "INVALID"


def metrics(gold, predicted, labels):
    correct = sum(a == b for a, b in zip(gold, predicted))
    per_label = {}
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(gold, predicted))
        fp = sum(a != label and b == label for a, b in zip(gold, predicted))
        fn = sum(a == label and b != label for a, b in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": sum(a == label for a in gold),
        }
    return {
        "examples": len(gold),
        "accuracy": correct / len(gold) if gold else 0.0,
        "macro_f1": sum(item["f1"] for item in per_label.values()) / len(labels),
        "invalid_predictions": Counter(predicted)["INVALID"],
        "per_label": per_label,
    }


def load_sst2():
    path = ROOT / "artifacts" / "datasets" / "sst2"
    if path.exists():
        return load_from_disk(str(path))
    dataset = load_dataset("stanfordnlp/sst2", "default")
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(path))
    return dataset


def stratified_sample(dataset):
    """Sample without replacement in proportion to the training label counts."""
    train = dataset["train"]
    validation_texts = set(dataset["validation"]["sentence"])
    buckets = {label: [] for label in range(len(train.features["label"].names))}
    for index, (text, label) in enumerate(zip(train["sentence"], train["label"])):
        if text not in validation_texts:
            buckets[label].append(index)
    available = sum(map(len, buckets.values()))
    if available < SAMPLE_SIZE:
        raise ValueError(f"Only {available} eligible training rows are available")

    # Largest-remainder allocation preserves the original class proportions.
    quotas = {label: SAMPLE_SIZE * len(indices) / available for label, indices in buckets.items()}
    counts = {label: int(quota) for label, quota in quotas.items()}
    remaining = SAMPLE_SIZE - sum(counts.values())
    for label in sorted(buckets, key=lambda label: (-(quotas[label] - counts[label]), label))[:remaining]:
        counts[label] += 1

    rng = random.Random(SAMPLE_SEED)
    selected = []
    for label in sorted(buckets):
        selected.extend(rng.sample(buckets[label], counts[label]))
    rng.shuffle(selected)
    return selected, counts


def open_checkpoint(path, settings):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            model TEXT NOT NULL,
            template_id TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            gold_label TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            PRIMARY KEY (model, template_id, row_index)
        )
    """)
    signature = json.dumps(settings, sort_keys=True)
    existing = conn.execute("SELECT value FROM metadata WHERE key = 'settings'").fetchone()
    if existing and existing[0] != signature:
        conn.close()
        raise ValueError(f"Checkpoint settings changed: {path}. Use a new --checkpoint path.")
    if not existing:
        conn.execute("INSERT INTO metadata VALUES ('settings', ?)", (signature,))
        conn.commit()
    return conn


def write_outputs(conn, checkpoint, output_dir, labels):
    records = conn.execute("""
        SELECT model, template_id, row_index, gold_label, predicted_label
        FROM predictions ORDER BY model, template_id, row_index
    """).fetchall()
    csv_path = output_dir / f"{checkpoint.stem}_predictions.csv"
    csv_temp = csv_path.with_suffix(".csv.tmp")
    with csv_temp.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "model", "dataset", "split", "template_id", "row_index",
            "gold_label", "predicted_label",
        ])
        writer.writerows(
            (model, "sst2", "train_sample_12000", template, index, gold, predicted)
            for model, template, index, gold, predicted in records
        )
    csv_temp.replace(csv_path)

    grouped = {}
    for model, template, _, gold, predicted in records:
        pair = grouped.setdefault((model, template), ([], []))
        pair[0].append(gold)
        pair[1].append(predicted)
    report = {"dataset": "sst2", "split": "train", "sample_size": SAMPLE_SIZE, "results": {}}
    for model in MODEL_IDS:
        report["results"][model] = {}
        for template in TEMPLATE_IDS:
            gold, predicted = grouped.get((model, template), ([], []))
            report["results"][model][template] = {
                "completed": len(gold),
                "expected": SAMPLE_SIZE,
                "complete": len(gold) == SAMPLE_SIZE,
                "metrics": metrics(gold, predicted, labels) if gold else None,
            }
    json_path = output_dir / f"{checkpoint.stem}_metrics.json"
    json_temp = json_path.with_suffix(".json.tmp")
    json_temp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    json_temp.replace(json_path)
    print(f"Saved {len(records)} predictions: {csv_path}", flush=True)
    print(f"Progress and metrics: {json_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--limit", type=int, help="Process the first N sampled rows per model/template")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but not available")

    dataset = load_sst2()
    selected, label_counts = stratified_sample(dataset)
    labels = dataset["train"].features["label"].names
    output_dir = ROOT / "artifacts"
    output_dir.mkdir(exist_ok=True)
    manifest = {
        "dataset": "stanfordnlp/sst2",
        "source_split": "train",
        "validation_rows_excluded": len(dataset["validation"]),
        "sample_size": SAMPLE_SIZE,
        "seed": SAMPLE_SEED,
        "label_counts": {labels[label]: count for label, count in label_counts.items()},
        "train_fingerprint": dataset["train"]._fingerprint,
        "validation_fingerprint": dataset["validation"]._fingerprint,
        "row_indices": selected,
    }
    manifest_path = output_dir / "sst2_train_12000_sample.json"
    manifest_text = json.dumps(manifest, indent=2)
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != manifest_text:
        raise ValueError(f"Sample manifest changed: {manifest_path}")
    if not manifest_path.exists():
        manifest_path.write_text(manifest_text, encoding="utf-8")

    settings = {
        "models": MODEL_IDS,
        "prompt_templates": dict(PROMPT_TEMPLATES["sst2"]),
        "sample_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        "decoding": {"do_sample": False, "max_new_tokens": 4},
        "device": device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    checkpoint = args.checkpoint or output_dir / f"sst2_train_12000_{device}.sqlite3"
    conn = open_checkpoint(checkpoint, settings)
    try:
        chosen = selected[:args.limit] if args.limit else selected
        for model_size in MODEL_IDS:
            work = []
            for template in TEMPLATE_IDS:
                completed = {
                    index for (index,) in conn.execute(
                        "SELECT row_index FROM predictions WHERE model = ? AND template_id = ?",
                        (model_size, template),
                    )
                }
                pending = [index for index in chosen if index not in completed]
                if pending:
                    work.append((template, pending))
            if not work:
                print(f"{model_size}: requested rows already checkpointed", flush=True)
                continue
            print(f"Loading {MODEL_IDS[model_size]} on {device}", flush=True)
            model, tokenizer = load_model(model_size, device)
            try:
                if model.device.type != device:
                    raise RuntimeError(f"Expected {device}, got {model.device}")
                print(f"{model_size} model device: {model.device}", flush=True)
                for template, pending in work:
                    print(f"{model_size} {template}: {len(pending)} rows remaining", flush=True)
                    for number, index in enumerate(pending, 1):
                        row = dataset["train"][index]
                        prediction = predict_label(model, tokenizer, template, row["sentence"], labels)
                        conn.execute(
                            "INSERT INTO predictions VALUES (?, ?, ?, ?, ?)",
                            (model_size, template, index, labels[row["label"]], prediction),
                        )
                        conn.commit()
                        if number % 100 == 0 or number == len(pending):
                            print(f"  {number}/{len(pending)} new rows saved", flush=True)
            finally:
                del model, tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print("Interrupted; completed predictions remain in the checkpoint.", flush=True)
    finally:
        write_outputs(conn, checkpoint, output_dir, labels)
        conn.close()


if __name__ == "__main__":
    main()
