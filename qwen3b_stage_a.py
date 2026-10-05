"""Checkpointed Stage A baseline for Qwen2.5-3B-Instruct on frozen examples."""

import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import torch
import transformers
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_templates import PROMPT_TEMPLATES, QWEN3B_SST2_T4_PROMPT

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_KEY = "3b"
LABELS = {"sst2": ("negative", "positive"),
          "tweet_eval_sentiment": ("negative", "neutral", "positive")}
EXPECTED = 4 * (12_872 + 12_284)
def effective_prompts():
    prompts = {name: dict(templates) for name, templates in PROMPT_TEMPLATES.items()}
    prompts["sst2"]["template_4"] = QWEN3B_SST2_T4_PROMPT
    return prompts


def render_prompt_3b(dataset, template, text):
    return effective_prompts()[dataset][template].format(text=text)


def load_frozen_rows(root):
    """Use prior completed Qwen scores only to fix row identities and gold labels."""
    source = root / "artifacts" / "pooled_stage_a" / "qwen_stage_a_pooled_scores.sqlite3"
    conn = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    try:
        rows = conn.execute("""SELECT dataset, split, template_id, row_index,
            source_split, source_row_index, gold_label FROM scores
            WHERE model='0.5b' ORDER BY dataset, template_id, row_index""").fetchall()
    finally:
        conn.close()
    if len(rows) != EXPECTED or len(set(row[:4] for row in rows)) != EXPECTED:
        raise ValueError(f"Frozen source incomplete or duplicated: {len(rows)}/{EXPECTED}")
    digest = hashlib.sha256()
    data = {name: load_from_disk(str(root / "artifacts" / "datasets" / name))
            for name in LABELS}
    work = []
    for dataset, split, template, row_index, source_split, source_index, gold in rows:
        if template not in PROMPT_TEMPLATES[dataset] or gold not in LABELS[dataset]:
            raise ValueError("Unexpected prompt or label in frozen source")
        if (dataset == "sst2" and split != "combined") or (
                dataset == "tweet_eval_sentiment" and (split != "test" or source_split != "test")):
            raise ValueError("Unexpected dataset split")
        if data[dataset][source_split][source_index]["label"] != LABELS[dataset].index(gold):
            raise ValueError("Gold label differs from frozen source")
        text_column = "sentence" if dataset == "sst2" else "text"
        item = (dataset, split, template, row_index, source_split, source_index,
                gold, data[dataset][source_split][source_index][text_column])
        digest.update(json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
        work.append(item)
    return work, digest.hexdigest(), {name: {split: part._fingerprint for split, part in ds.items()}
                                   for name, ds in data.items()}


def open_checkpoint(path, signature, overwrite_3b=False, replace_sst2_t4=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("""CREATE TABLE IF NOT EXISTS scores(
        model TEXT NOT NULL, dataset TEXT NOT NULL, split TEXT NOT NULL,
        template_id TEXT NOT NULL, row_index INTEGER NOT NULL,
        gold_label TEXT NOT NULL, predicted_label TEXT NOT NULL,
        logp_positive REAL NOT NULL, logp_negative REAL NOT NULL,
        logp_neutral REAL, margin REAL NOT NULL,
        source_split TEXT NOT NULL, source_row_index INTEGER NOT NULL,
        PRIMARY KEY(model,dataset,split,template_id,row_index))""")
    saved = conn.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
    wanted = json.dumps(signature, sort_keys=True)
    if replace_sst2_t4 and saved and saved[0] != wanted:
        previous = json.loads(saved[0])
        expected_previous = json.loads(wanted)
        expected_previous["prompts"]["sst2"]["template_4"] = (
            "Classify the sentiment as exactly one label: positive or negative. "
            "Return only POSITIVE OR NEGATIVE LABEL. No explanation.\n\n"
            "Text: {text}"
        )
        if previous != expected_previous:
            conn.close()
            raise ValueError("Checkpoint differs beyond the authorized SST-2 T4 wording")
        total = conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        t4_count = conn.execute("""SELECT COUNT(*) FROM scores WHERE model='3b'
            AND dataset='sst2' AND split='combined' AND template_id='template_4'""").fetchone()[0]
        if total != EXPECTED or t4_count != 12_872:
            conn.close()
            raise ValueError(f"Expected complete old checkpoint; found {total} total, {t4_count} SST-2 T4")
        with conn:
            deleted = conn.execute("""DELETE FROM scores WHERE model='3b'
                AND dataset='sst2' AND split='combined' AND template_id='template_4'""").rowcount
            if deleted != t4_count:
                raise ValueError("SST-2 T4 deletion count changed")
            conn.execute("UPDATE metadata SET value=? WHERE key='signature'", (wanted,))
        print(f"Removed {deleted:,} previous SST-2 T4 rows from the existing checkpoint", flush=True)
    elif overwrite_3b:
        if saved:
            previous = json.loads(saved[0])
            for key in ("model_id", "source_sha256", "dataset_fingerprints", "candidate_ids"):
                if previous.get(key) != signature[key]:
                    conn.close()
                    raise ValueError(f"Cannot overwrite checkpoint: {key} changed")
        models = {row[0] for row in conn.execute("SELECT DISTINCT model FROM scores")}
        if models - {MODEL_KEY}:
            conn.close()
            raise ValueError(f"Checkpoint contains other models: {models}")
        with conn:
            deleted = conn.execute("DELETE FROM scores WHERE model=?", (MODEL_KEY,)).rowcount
            if saved:
                conn.execute("UPDATE metadata SET value=? WHERE key='signature'", (wanted,))
            else:
                conn.execute("INSERT INTO metadata VALUES('signature',?)", (wanted,))
        print(f"Removed {deleted:,} old 3B rows from the existing checkpoint", flush=True)
    elif saved and saved[0] != wanted:
        conn.close()
        raise ValueError("Checkpoint signature changed; use --overwrite-3b for the requested full replacement")
    elif not saved:
        conn.execute("INSERT INTO metadata VALUES('signature',?)", (wanted,))
        conn.commit()
    return conn


def score_and_generate(model, tokenizer, items, candidate_ids):
    chats = [tokenizer.apply_chat_template(
        [{"role": "user", "content": render_prompt_3b(row[0], row[2], row[7])}],
        tokenize=False, add_generation_prompt=True) for row in items]
    batch = tokenizer(chats, padding=True, truncation=True, max_length=1024,
                      return_tensors="pt").to("cuda")
    with torch.inference_mode():
        logits = model(**batch, use_cache=False).logits[:, -1, :].float()
        log_z = torch.logsumexp(logits, dim=-1)
        logps = {label: (logits[:, token_id] - log_z).cpu().tolist()
                for label, token_id in candidate_ids.items()}
        outputs = model.generate(**batch, do_sample=False, temperature=0.0,
                                 max_new_tokens=4,
                                 pad_token_id=tokenizer.eos_token_id)
    answers = tokenizer.batch_decode(outputs[:, batch.input_ids.shape[1]:],
                                     skip_special_tokens=True)
    result = []
    for i, row in enumerate(items):
        answer = answers[i].strip().lower().strip(" .,!?:;\"'`\n\t")
        predicted = answer if answer in LABELS[row[0]] else "INVALID"
        pos, neg = logps["positive"][i], logps["negative"][i]
        neutral = logps["neutral"][i] if row[0] == "tweet_eval_sentiment" else None
        result.append((MODEL_KEY, *row[:4], row[6], predicted, pos, neg, neutral,
                       pos - neg, row[4], row[5]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, help="First N examples per dataset/template")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "artifacts" / "qwen3b_stage_a" / "scores.sqlite3")
    replacement = parser.add_mutually_exclusive_group()
    replacement.add_argument("--overwrite-3b", action="store_true",
                             help="Remove all existing 3B rows and rerun in this same checkpoint")
    replacement.add_argument("--replace-sst2-t4", action="store_true",
                             help="Replace only completed 3B SST-2 T4 rows in this checkpoint")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch-size and limit must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    print(f"CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    work, source_hash, fingerprints = load_frozen_rows(ROOT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    candidate_ids = {}
    for label in ("positive", "negative", "neutral"):
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Candidate {label} is not a single token: {ids}")
        candidate_ids[label] = ids[0]
    signature = {"model_id": MODEL_ID, "source_sha256": source_hash,
                 "dataset_fingerprints": fingerprints,
                 "prompts": effective_prompts(),
                 "candidate_ids": candidate_ids, "greedy_new_tokens": 4,
                 "do_sample": False, "temperature": 0.0,
                 "score": "bare lowercase label at first assistant token, log-softmax",
                 "dtype": "float16", "torch": torch.__version__,
                 "transformers": transformers.__version__}
    conn = open_checkpoint(args.checkpoint.resolve(), signature,
                           overwrite_3b=args.overwrite_3b,
                           replace_sst2_t4=args.replace_sst2_t4)
    try:
        done = set(conn.execute("SELECT dataset,split,template_id,row_index FROM scores"))
        selected = []
        group_count = {}
        for row in work:
            group = row[:3]
            group_count[group] = group_count.get(group, 0) + 1
            if (args.limit is None or group_count[group] <= args.limit) and row[:4] not in done:
                selected.append(row)
        print(f"Checkpoint: {len(done):,}/{EXPECTED:,}; pending: {len(selected):,}", flush=True)
        if selected:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, dtype=torch.float16, local_files_only=True).eval().to("cuda")
            if model.device.type != "cuda":
                raise RuntimeError("Model not on CUDA")
            print(f"Model: {MODEL_ID}, device: {model.device}, candidate IDs: {candidate_ids}", flush=True)
            for start in range(0, len(selected), args.batch_size):
                rows = score_and_generate(model, tokenizer,
                                          selected[start:start + args.batch_size], candidate_ids)
                with conn:
                    conn.executemany("INSERT INTO scores VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
                count = start + len(rows)
                if count % 500 < args.batch_size or count == len(selected):
                    print(f"Saved {len(done) + count:,}/{EXPECTED:,}", flush=True)
        total = conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        print(f"Checkpoint now has {total:,}/{EXPECTED:,}; complete={total == EXPECTED}", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
