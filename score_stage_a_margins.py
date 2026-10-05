"""Score fixed sentiment label tokens on CUDA using existing Stage A predictions.

No text generation occurs. Each batch saves log P(positive), log P(negative),
their difference, and (for TweetEval) log P(neutral) to a resumable checkpoint.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import torch
import transformers
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_templates import PROMPT_TEMPLATES, render_prompt

MODELS = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}
GROUPS = (
    ("sst2", "train", 12_000, "sentence"),
    ("sst2", "validation", 872, "sentence"),
    ("tweet_eval_sentiment", "test", 12_284, "text"),
)
LABELS = ("positive", "negative", "neutral")


def read_source_rows(artifacts):
    """Read exactly the completed, labeled examples from the two prior runs."""
    groups = defaultdict(list)
    source_hash = hashlib.sha256()
    sources = (
        (artifacts / "all_templates_cuda.sqlite3", "full"),
        (artifacts / "sst2_train_12000_cuda.sqlite3", "train_sample"),
    )
    for path, kind in sources:
        if not path.is_file():
            raise FileNotFoundError(f"Required prediction checkpoint is missing: {path}")
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            if kind == "full":
                rows = conn.execute("""
                    SELECT model, dataset, split, template_id, row_index,
                           gold_label, predicted_label
                    FROM predictions
                    ORDER BY model, dataset, split, template_id, row_index
                """)
            else:
                rows = conn.execute("""
                    SELECT model, 'sst2', 'train', template_id, row_index,
                           gold_label, predicted_label
                    FROM predictions
                    ORDER BY model, template_id, row_index
                """)
            for record in rows:
                model, dataset, split, template, index, gold, predicted = record
                if model not in MODELS or dataset not in PROMPT_TEMPLATES:
                    raise ValueError(f"Unexpected source record: {record}")
                groups[(model, dataset, split, template)].append(
                    (index, gold, predicted)
                )
                source_hash.update(json.dumps(record, separators=(",", ":")).encode("utf-8"))
                source_hash.update(b"\n")
        finally:
            conn.close()
    expected_groups = len(MODELS) * len(PROMPT_TEMPLATES["sst2"]) * len(GROUPS)
    if len(groups) != expected_groups:
        raise ValueError(f"Expected {expected_groups} result groups, found {len(groups)}")
    for model in MODELS:
        for dataset, split, expected, _ in GROUPS:
            for template in PROMPT_TEMPLATES[dataset]:
                key = (model, dataset, split, template)
                rows = groups[key]
                if len(rows) != expected or len({row[0] for row in rows}) != expected:
                    raise ValueError(f"Incomplete or duplicate source group {key}: {len(rows)}/{expected}")
    return groups, source_hash.hexdigest()


def open_checkpoint(path, settings):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            model TEXT NOT NULL,
            dataset TEXT NOT NULL,
            split TEXT NOT NULL,
            template_id TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            gold_label TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            logp_positive REAL NOT NULL,
            logp_negative REAL NOT NULL,
            logp_neutral REAL,
            margin REAL NOT NULL,
            PRIMARY KEY (model, dataset, split, template_id, row_index)
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


def candidate_ids(tokenizer):
    ids = {}
    for label in LABELS:
        token_ids = tokenizer.encode(label, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"'{label}' is not one token for this tokenizer: {token_ids}")
        ids[label] = token_ids[0]
    return ids


def score_batch(model, tokenizer, dataset, template, texts, ids):
    chats = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": render_prompt(dataset, template, text)}],
            tokenize=False, add_generation_prompt=True,
        )
        for text in texts
    ]
    inputs = tokenizer(
        chats, return_tensors="pt", padding=True,
        truncation=True, max_length=1024,
    ).to("cuda")
    if inputs["input_ids"].device.type != "cuda":
        raise RuntimeError("Input tensors are not on CUDA")
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits[:, -1, :].float()
        log_z = torch.logsumexp(logits, dim=-1)
        positive = logits[:, ids["positive"]] - log_z
        negative = logits[:, ids["negative"]] - log_z
        neutral = logits[:, ids["neutral"]] - log_z if dataset == "tweet_eval_sentiment" else None
        margin = positive - negative
    positive_values = positive.cpu().tolist()
    negative_values = negative.cpu().tolist()
    margin_values = margin.cpu().tolist()
    neutral_values = neutral.cpu().tolist() if neutral is not None else [None] * len(texts)
    return list(zip(positive_values, negative_values, neutral_values, margin_values))


def export(conn, path, total_expected):
    csv_path = path.with_suffix(".csv")
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "model", "dataset", "split", "template_id", "row_index",
            "gold_label", "predicted_label", "logp_positive", "logp_negative",
            "logp_neutral", "margin",
        ])
        writer.writerows(conn.execute("""
            SELECT model, dataset, split, template_id, row_index,
                   gold_label, predicted_label, logp_positive, logp_negative,
                   logp_neutral, margin
            FROM scores ORDER BY model, dataset, split, template_id, row_index
        """))
    temporary.replace(csv_path)
    completed = conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
    progress_path = path.with_name(path.stem + "_progress.json")
    temporary_progress = progress_path.with_suffix(".json.tmp")
    temporary_progress.write_text(json.dumps({
        "checkpoint": str(path), "completed": completed,
        "expected": total_expected, "complete": completed == total_expected,
        "csv": str(csv_path),
    }, indent=2), encoding="utf-8")
    temporary_progress.replace(progress_path)
    print(f"Scores saved: {completed}/{total_expected}; CSV: {csv_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, help="Score at most N rows per group for a smoke run")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--batch-size and --limit must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; refusing to score on CPU")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    artifacts = ROOT / "artifacts"
    source, source_hash = read_source_rows(artifacts)
    datasets = {
        name: load_from_disk(str(artifacts / "datasets" / name))
        for name in ("sst2", "tweet_eval_sentiment")
    }
    text_columns = {
        (name, split): datasets[name][split][column]
        for name, split, _, column in GROUPS
    }
    settings = {
        "models": MODELS,
        "prompts": {name: dict(prompts) for name, prompts in PROMPT_TEMPLATES.items()},
        "source_sha256": source_hash,
        "dataset_fingerprints": {
            f"{name}:{split}": datasets[name][split]._fingerprint
            for name, split, _, _ in GROUPS
        },
        "score_definition": "log softmax of bare lowercase label token at first assistant position",
        "margin_definition": "logp_positive - logp_negative",
        "dtype": "float16 model / float32 logits",
        "device": "cuda",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    checkpoint = args.checkpoint or artifacts / "stage_a_margins_cuda.sqlite3"
    conn = open_checkpoint(checkpoint, settings)
    total_expected = sum(len(rows) for rows in source.values())
    try:
        for model_size, model_id in MODELS.items():
            work = []
            for dataset, split, _, _ in GROUPS:
                for template in PROMPT_TEMPLATES[dataset]:
                    key = (model_size, dataset, split, template)
                    rows = source[key]
                    selected = rows[:args.limit] if args.limit else rows
                    done = {
                        index for (index,) in conn.execute("""
                            SELECT row_index FROM scores
                            WHERE model = ? AND dataset = ? AND split = ? AND template_id = ?
                        """, key)
                    }
                    pending = [row for row in selected if row[0] not in done]
                    if pending:
                        work.append((dataset, split, template, pending))
            if not work:
                print(f"{model_size}: requested scores already checkpointed", flush=True)
                continue
            print(f"Loading {model_id} on CUDA", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            ids = candidate_ids(tokenizer)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.float16, device_map="cuda",
                local_files_only=True,
            ).eval()
            try:
                if model.device.type != "cuda":
                    raise RuntimeError(f"Model loaded on {model.device}, expected CUDA")
                print(f"{model_size} model device: {model.device}; label tokens: {ids}", flush=True)
                for dataset, split, template, pending in work:
                    print(f"{model_size} {dataset}/{split} {template}: {len(pending)} remaining", flush=True)
                    texts = text_columns[(dataset, split)]
                    for start in range(0, len(pending), args.batch_size):
                        batch = pending[start:start + args.batch_size]
                        values = score_batch(
                            model, tokenizer, dataset, template,
                            [texts[index] for index, _, _ in batch], ids,
                        )
                        conn.executemany("""
                            INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, [
                            (model_size, dataset, split, template, index, gold, predicted,
                             logp_pos, logp_neg, logp_neu, margin)
                            for (index, gold, predicted),
                                (logp_pos, logp_neg, logp_neu, margin)
                            in zip(batch, values)
                        ])
                        conn.commit()
                        count = min(start + len(batch), len(pending))
                        if count % 1024 < args.batch_size or count == len(pending):
                            print(f"  {count}/{len(pending)} new scores saved", flush=True)
            finally:
                del model, tokenizer
                gc.collect()
                torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print("Interrupted; committed batches remain in the checkpoint.", flush=True)
    finally:
        export(conn, checkpoint, total_expected)
        conn.close()


if __name__ == "__main__":
    main()
