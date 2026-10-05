"""Run every fixed sentiment prompt with per-prediction SQLite checkpoints."""

import argparse
import csv
import json
import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import torch
import transformers

from evaluate_sentiment import TASKS, metrics, predict_label
from load_assets import MODEL_IDS, load_model, load_sentiment_datasets
from prompt_templates import PROMPT_TEMPLATES


TEMPLATE_IDS = tuple(PROMPT_TEMPLATES["sst2"])


def setup_checkpoint(path, settings):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            model TEXT NOT NULL,
            dataset TEXT NOT NULL,
            template_id TEXT NOT NULL,
            split TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            gold_label TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            PRIMARY KEY (model, dataset, template_id, row_index)
        )
    """)
    signature = json.dumps(settings, sort_keys=True)
    existing = conn.execute(
        "SELECT value FROM metadata WHERE key = 'settings'"
    ).fetchone()
    if existing and existing[0] != signature:
        conn.close()
        raise ValueError(
            f"Checkpoint settings differ: {path}. Use a new --checkpoint path "
            "to start a separate run."
        )
    if not existing:
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES ('settings', ?)", (signature,)
        )
        conn.commit()
    return conn


def export_results(conn, datasets, output_dir, checkpoint_path):
    rows = conn.execute("""
        SELECT model, dataset, template_id, split, row_index,
               gold_label, predicted_label
        FROM predictions
        ORDER BY model, dataset, template_id, row_index
    """).fetchall()
    csv_path = output_dir / f"{checkpoint_path.stem}_predictions.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "model", "dataset", "template_id", "split", "row_index",
            "gold_label", "predicted_label",
        ])
        writer.writerows(rows)
    temporary_csv.replace(csv_path)

    grouped = {}
    for model, dataset_name, template_id, split, index, gold, predicted in rows:
        grouped.setdefault((model, dataset_name, template_id), ([], []))
        grouped[(model, dataset_name, template_id)][0].append(gold)
        grouped[(model, dataset_name, template_id)][1].append(predicted)
    report = {"checkpoint": str(checkpoint_path), "results": {}}
    for model_size in MODEL_IDS:
        report["results"][model_size] = {}
        for dataset_name, (split, _) in TASKS.items():
            labels = datasets[dataset_name]["train"].features["label"].names
            expected = sum(label >= 0 for label in datasets[dataset_name][split]["label"])
            report["results"][model_size][dataset_name] = {}
            for template_id in TEMPLATE_IDS:
                gold, predicted = grouped.get((model_size, dataset_name, template_id), ([], []))
                result = metrics(gold, predicted, labels) if gold else None
                report["results"][model_size][dataset_name][template_id] = {
                    "completed": len(gold),
                    "expected": expected,
                    "complete": len(gold) == expected,
                    "metrics": result,
                }
    json_path = output_dir / f"{checkpoint_path.stem}_metrics.json"
    temporary_json = json_path.with_suffix(".json.tmp")
    temporary_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary_json.replace(json_path)
    print(f"Saved {len(rows)} predictions to {csv_path}", flush=True)
    print(f"Progress and metrics: {json_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Process only the first N rows of each split")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--checkpoint", type=Path, help="SQLite checkpoint path")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but PyTorch cannot detect a CUDA device")
    datasets = load_sentiment_datasets()
    settings = {
        "models": MODEL_IDS,
        "tasks": TASKS,
        "prompts": {name: dict(templates) for name, templates in PROMPT_TEMPLATES.items()},
        "dataset_fingerprints": {
            name: datasets[name][split]._fingerprint for name, (split, _) in TASKS.items()
        },
        "decoding": {"do_sample": False, "max_new_tokens": 4},
        "device": device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    output_dir = ROOT / "artifacts"
    checkpoint_path = args.checkpoint or output_dir / f"all_templates_{device}.sqlite3"
    conn = setup_checkpoint(checkpoint_path, settings)
    try:
        for model_size in MODEL_IDS:
            work = []
            for dataset_name, (split, text_column) in TASKS.items():
                rows = datasets[dataset_name][split]
                count = min(len(rows), args.limit) if args.limit else len(rows)
                for template_id in TEMPLATE_IDS:
                    completed = {
                        index for (index,) in conn.execute("""
                            SELECT row_index FROM predictions
                            WHERE model = ? AND dataset = ? AND template_id = ?
                        """, (model_size, dataset_name, template_id))
                    }
                    pending = [
                        index for index in range(count)
                        if index not in completed and rows[index]["label"] >= 0
                    ]
                    if pending:
                        work.append((dataset_name, split, text_column, template_id, pending))
            if not work:
                print(f"{model_size}: requested rows already checkpointed", flush=True)
                continue
            print(f"Loading {MODEL_IDS[model_size]} on {device}", flush=True)
            model, tokenizer = load_model(model_size, device)
            try:
                for dataset_name, split, text_column, template_id, pending in work:
                    rows = datasets[dataset_name][split]
                    labels = datasets[dataset_name]["train"].features["label"].names
                    print(
                        f"{model_size} {dataset_name} {template_id}: "
                        f"{len(pending)} rows remaining", flush=True,
                    )
                    for number, index in enumerate(pending, 1):
                        row = rows[index]
                        prediction = predict_label(
                            model, tokenizer, dataset_name, template_id,
                            row[text_column], labels,
                        )
                        conn.execute("""
                            INSERT INTO predictions
                            (model, dataset, template_id, split, row_index,
                             gold_label, predicted_label)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            model_size, dataset_name, template_id, split, index,
                            labels[row["label"]], prediction,
                        ))
                        conn.commit()  # Each completed prediction is a checkpoint.
                        if number % 100 == 0 or number == len(pending):
                            print(f"  {number}/{len(pending)} new rows saved", flush=True)
            finally:
                del model, tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print("Interrupted; completed predictions are safe in the checkpoint.", flush=True)
    finally:
        export_results(conn, datasets, output_dir, checkpoint_path)
        conn.close()


if __name__ == "__main__":
    main()
