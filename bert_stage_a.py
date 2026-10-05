"""CUDA-only, checkpointed zero-shot sentiment evaluation with BERT-Large MLM.

Use the same pooled SST-2 rows and TweetEval test rows as Qwen Stage A.
Each fixed cloze prompt has one [MASK]; the highest-scoring allowed label wins.
"""

import argparse
import csv
import hashlib
import json
import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import torch
from datasets import load_from_disk
from transformers import AutoModelForMaskedLM, AutoTokenizer


MODEL_ID = "google-bert/bert-large-uncased"
TEMPLATES = {
    "template_1": "The sentiment is [MASK]. Text: {text}",
    "template_2": "This text has a [MASK] tone. Text: {text}",
    "template_3": "A [MASK] opinion is expressed here: {text}",
    "template_4": "Sentiment label: [MASK]. Review: {text}",
}
LABELS = {
    "sst2": ("negative", "positive"),
    "tweet_eval_sentiment": ("negative", "neutral", "positive"),
}


def load_rows():
    pooled = load_from_disk(str(ROOT / "artifacts" / "pooled_stage_a" / "sst2_combined_12872"))
    tweets = load_from_disk(str(ROOT / "artifacts" / "datasets" / "tweet_eval_sentiment"))["test"]
    if len(pooled) != 12872 or len(tweets) != 12284:
        raise ValueError("Dataset row counts changed; refusing to mix checkpoints")
    if list(pooled.features["label"].names) != list(LABELS["sst2"]):
        raise ValueError("Unexpected SST-2 label mapping")
    if list(tweets.features["label"].names) != list(LABELS["tweet_eval_sentiment"]):
        raise ValueError("Unexpected TweetEval label mapping")
    result = {}
    for dataset, source in (("sst2", pooled), ("tweet_eval_sentiment", tweets)):
        rows = []
        for i, row in enumerate(source):
            if dataset == "sst2":
                if row["combined_index"] != i:
                    raise ValueError("Pooled SST-2 order changed")
                split, source_index = row["source_split"], row["source_row_index"]
            else:
                split, source_index = "test", i
            rows.append((i, row["text"], LABELS[dataset][row["label"]], split, source_index))
        result[dataset] = rows
    return result


def source_digest(rows):
    digest = hashlib.sha256()
    for dataset, dataset_rows in rows.items():
        for row in dataset_rows:
            digest.update(json.dumps((dataset, row), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def open_checkpoint(path, signature):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("""CREATE TABLE IF NOT EXISTS scores (
        dataset TEXT NOT NULL, template_id TEXT NOT NULL, row_index INTEGER NOT NULL,
        source_split TEXT NOT NULL, source_row_index INTEGER NOT NULL,
        gold_label TEXT NOT NULL, predicted_label TEXT NOT NULL,
        logp_positive REAL NOT NULL, logp_negative REAL NOT NULL,
        logp_neutral REAL, margin REAL NOT NULL,
        PRIMARY KEY(dataset, template_id, row_index))""")
    old = conn.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
    signature = json.dumps(signature, sort_keys=True)
    if old and old[0] != signature:
        conn.close()
        raise ValueError("Checkpoint inputs or prompts changed. Use a new --checkpoint path.")
    if not old:
        conn.execute("INSERT INTO metadata VALUES ('signature', ?)", (signature,))
        conn.commit()
    return conn


def metrics(conn, output_dir):
    summary = []
    for dataset, labels in LABELS.items():
        for template_id in TEMPLATES:
            counts = {(gold, pred): 0 for gold in labels for pred in labels}
            for gold, pred in conn.execute(
                "SELECT gold_label, predicted_label FROM scores WHERE dataset=? AND template_id=?",
                (dataset, template_id),
            ):
                counts[(gold, pred)] += 1
            n = sum(counts.values())
            if not n:
                continue
            accuracy = sum(counts[(label, label)] for label in labels) / n
            f1, recall = [], []
            for label in labels:
                tp = counts[(label, label)]
                actual = sum(counts[(label, pred)] for pred in labels)
                predicted = sum(counts[(gold, label)] for gold in labels)
                f1.append(2 * tp / (actual + predicted) if actual + predicted else 0)
                recall.append(tp / actual if actual else 0)
            summary.append((dataset, template_id, n, accuracy, sum(f1) / len(f1), sum(recall) / len(recall)))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "performance_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "dataset", "template_id", "examples", "accuracy", "macro_f1", "macro_recall"))
        for row in summary:
            writer.writerow((MODEL_ID, *row))
    with (output_dir / "results.txt").open("w", encoding="utf-8") as handle:
        handle.write("BERT-Large-Uncased masked-LM zero-shot sentiment baseline\n")
        handle.write("SST-2 = pooled 12,000 train + 872 validation; TweetEval = test 12,284.\n")
        handle.write("SST-2 pooled score is not the official validation score.\n")
        handle.write("BERT cloze scores and Qwen generative scores use different objectives.\n\n")
        for dataset, template_id, n, accuracy, macro_f1, macro_recall in summary:
            handle.write(f"{dataset} {template_id} n={n}: accuracy={accuracy:.4f} macro_F1={macro_f1:.4f} macro_recall={macro_recall:.4f}\n")
    return summary


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; BERT will not silently fall back to CPU")
    print(f"CUDA: {torch.cuda.get_device_name(0)}; torch {torch.__version__}", flush=True)
    rows = load_rows()
    signature = {"model": MODEL_ID, "templates": TEMPLATES, "labels": LABELS,
                 "source_sha256": source_digest(rows), "max_length": args.max_length}
    conn = open_checkpoint(args.checkpoint, signature)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        label_ids = {}
        for label in ("negative", "neutral", "positive"):
            ids = tokenizer.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Expected one WordPiece token for {label}, got {ids}")
            label_ids[label] = ids[0]
        print(f"Label token IDs: {label_ids}", flush=True)
        model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, dtype=torch.float16).eval().to("cuda")
        print(f"Model parameter device: {next(model.parameters()).device}", flush=True)
        scored = 0
        for dataset, dataset_rows in rows.items():
            for template_id, template in TEMPLATES.items():
                completed = {r[0] for r in conn.execute(
                    "SELECT row_index FROM scores WHERE dataset=? AND template_id=?", (dataset, template_id))}
                pending = [r for r in dataset_rows if r[0] not in completed]
                if args.limit is not None:
                    pending = [r for r in pending if r[0] < args.limit]
                print(f"{dataset} {template_id}: {len(completed)} saved, {len(pending)} pending", flush=True)
                for start in range(0, len(pending), args.batch_size):
                    batch = pending[start:start + args.batch_size]
                    prompts = [template.format(text=row[1]) for row in batch]
                    inputs = tokenizer(prompts, padding=True, truncation=True,
                                       max_length=args.max_length, return_tensors="pt").to("cuda")
                    mask_positions = inputs["input_ids"].eq(tokenizer.mask_token_id)
                    if not torch.all(mask_positions.sum(dim=1) == 1).item():
                        raise ValueError("A prompt has zero or multiple [MASK] tokens")
                    with torch.inference_mode():
                        logits = model(**inputs).logits[mask_positions].float()
                        candidate_ids = [label_ids[label] for label in LABELS[dataset]]
                        candidate_logits = logits[:, candidate_ids]
                        # Full-vocabulary probabilities define comparable log P for the margin.
                        logp = torch.log_softmax(logits, dim=-1)
                        predictions = candidate_logits.argmax(dim=-1).tolist()
                        selected = logp[:, [label_ids[x] for x in ("positive", "negative", "neutral")]].cpu().tolist()
                    records = []
                    for row, prediction, scores in zip(batch, predictions, selected):
                        index, _, gold, split, source_index = row
                        lp_pos, lp_neg, lp_neu = scores
                        records.append((dataset, template_id, index, split, source_index, gold,
                                        LABELS[dataset][prediction], lp_pos, lp_neg,
                                        lp_neu if dataset == "tweet_eval_sentiment" else None,
                                        lp_pos - lp_neg))
                    with conn:
                        conn.executemany("INSERT INTO scores VALUES (?,?,?,?,?,?,?,?,?,?,?)", records)
                    scored += len(batch)
                    if (start // args.batch_size) % 25 == 0:
                        print(f"  saved {start + len(batch)}/{len(pending)}", flush=True)
        summary = metrics(conn, args.checkpoint.parent)
        print(f"Saved {scored} new rows; {len(summary)} metric groups", flush=True)
        for dataset, template, n, accuracy, f1, recall in summary:
            print(f"{dataset} {template}: n={n}, accuracy={accuracy:.4f}, macro-F1={f1:.4f}, macro-recall={recall:.4f}")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "bert_stage_a" / "scores.sqlite3")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--limit", type=int, help="Smoke-test first N rows per dataset/template, then resume without it")
    arguments = parser.parse_args()
    if arguments.batch_size < 1 or arguments.max_length < 16 or (arguments.limit is not None and arguments.limit < 1):
        parser.error("batch-size >= 1, max-length >= 16, and limit >= 1 are required")
    run(arguments)
