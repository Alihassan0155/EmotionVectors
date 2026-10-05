"""B3-B6: checkpointed NNsight residual extraction and raw emotion vectors.

Only strict single-label GoEmotions vector_train rows are used. This script
does not use validation, test, SST-2, or TweetEval. Neutral-PCA denoising is
reserved for B7-B8; directions saved here are explicitly raw contrasts.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import nnsight
import numpy as np
import torch
import transformers
from datasets import load_from_disk
from nnsight import NNsight
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare_goemotions_b2 import split_digest


MODELS = {"0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
          "1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
          "3b": "Qwen/Qwen2.5-3B-Instruct"}
EMOTIONS = ("joy", "love", "excitement", "gratitude", "sadness", "anger", "fear", "disgust")
MAX_TOKENS = 1024


def dataset_digest(dataset):
    digest = hashlib.sha256()
    for index, row in enumerate(dataset):
        record = (index, row["id"], row["text"], row["emotion"],
                  row["source_split"], row["source_row_index"], row["labels"])
        digest.update(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_training():
    root = ROOT / "artifacts" / "goemotions_b2"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    data = load_from_disk(str(root / "vector_train"))
    if len(data) != 7417 or set(data["emotion"]) != set(EMOTIONS):
        raise ValueError("Unexpected strict single-label training data")
    if any(row["source_split"] != "train" or len(row["labels"]) != 1 for row in data):
        raise ValueError("A training row is not a strict single-label train example")
    # The B2 manifest uses its own row digest; the full source digest below
    # additionally binds input text and order to this checkpoint.
    if manifest["counts"]["vector_train"] != len(data):
        raise ValueError("B2 manifest and training data disagree")
    if split_digest(data) != manifest["sha256"]["vector_train"]:
        raise ValueError("B2 training rows differ from their saved checksum")
    return data, dataset_digest(data), manifest["sha256"]["vector_train"]


def checkpoint(output, signature, nrows, layers, width):
    output.mkdir(parents=True, exist_ok=True)
    db_path = output / "progress.sqlite3"
    array_path = output / "pooled.float16.npy"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
    conn.execute("""CREATE TABLE IF NOT EXISTS completed(
        row_index INTEGER PRIMARY KEY, emotion TEXT NOT NULL,
        source_id TEXT NOT NULL, token_count INTEGER NOT NULL,
        text_sha256 TEXT NOT NULL)""")
    expected = json.dumps(signature, sort_keys=True)
    old = conn.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
    if old and old[0] != expected:
        conn.close()
        raise ValueError(f"Checkpoint settings differ: {output}. Choose a new --output.")
    if not old:
        conn.execute("INSERT INTO metadata VALUES ('signature',?)", (expected,))
        conn.commit()
    saved = conn.execute("SELECT COUNT(*) FROM completed").fetchone()[0]
    if array_path.exists():
        pooled = np.load(array_path, mmap_mode="r+")
        if pooled.shape != (nrows, layers, width) or pooled.dtype != np.float16:
            conn.close()
            raise ValueError("Pooled array shape or dtype differs from model")
    else:
        if saved:
            conn.close()
            raise ValueError("Checkpoint has completed rows but no pooled array")
        pooled = np.lib.format.open_memmap(array_path, mode="w+", dtype=np.float16,
                                           shape=(nrows, layers, width))
    return conn, pooled


def tokenized_batch(tokenizer, batch):
    encoded = tokenizer([row["text"] for row in batch], add_special_tokens=False,
                        padding=True, truncation=False, return_tensors="pt")
    if encoded["input_ids"].shape[1] > MAX_TOKENS:
        raise ValueError("An example exceeds MAX_TOKENS; no truncation is allowed")
    special = torch.zeros_like(encoded["attention_mask"], dtype=torch.bool)
    for token_id in tokenizer.all_special_ids:
        special |= encoded["input_ids"].eq(token_id)
    text_mask = encoded["attention_mask"].bool() & ~special
    lengths = text_mask.sum(dim=1)
    if (lengths == 0).any().item():
        raise ValueError("An example has zero valid text tokens")
    return encoded.to("cuda"), text_mask.to("cuda"), lengths.tolist()


def extract_pooled(wrapped, encoded, text_mask, layers):
    mask = text_mask.unsqueeze(-1)
    denominator = mask.sum(dim=1)
    with torch.inference_mode():
        with wrapped.trace(encoded["input_ids"],
                           attention_mask=encoded["attention_mask"], use_cache=False):
            pooled_layers = []
            for layer_index in range(layers):
                residual = wrapped.model.layers[layer_index].output
                pooled_layers.append((residual.float() * mask).sum(dim=1) / denominator)
            pooled = torch.stack(pooled_layers, dim=1).cpu().save()
    if pooled.ndim != 3 or pooled.shape[1] != layers or not torch.isfinite(pooled).all().item():
        raise ValueError("Invalid pooled activation shape or values")
    return pooled.numpy().astype(np.float16)


def export_vectors(data, pooled, output, model_id, signature,
                   layer_axis="decoder_layer_1_based"):
    means = np.empty((len(EMOTIONS), pooled.shape[1], pooled.shape[2]), dtype=np.float32)
    counts = {}
    for emotion_index, emotion in enumerate(EMOTIONS):
        indices = [index for index, row in enumerate(data) if row["emotion"] == emotion]
        counts[emotion] = len(indices)
        if not indices:
            raise ValueError(f"No examples for {emotion}")
        total = np.zeros(pooled.shape[1:], dtype=np.float64)
        for start in range(0, len(indices), 128):
            total += np.asarray(pooled[indices[start:start + 128]], dtype=np.float64).sum(axis=0)
        means[emotion_index] = (total / len(indices)).astype(np.float32)
    other = (means.astype(np.float64).sum(axis=0, keepdims=True) - means) / (len(EMOTIONS) - 1)
    raw = (means - other).astype(np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite raw emotion vector")
    np.save(output / "class_means.float32.npy", means)
    np.save(output / "emotion_directions_raw.float32.npy", raw)
    with (output / "vector_norms.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "layer_1_based", "emotion", "train_examples", "raw_l2_norm"))
        for layer in range(raw.shape[1]):
            for index, emotion in enumerate(EMOTIONS):
                writer.writerow((model_id, layer + 1, emotion, counts[emotion],
                                 float(np.linalg.norm(raw[index, layer]))))
    info = {**signature, "complete": True, "examples": len(data),
            "emotion_order": EMOTIONS, "class_counts": counts,
            "array_axes": ["emotion", layer_axis, "hidden_dimension"],
            "raw_vector_formula": "mu_e - mean(mu_other_7_emotions)",
            "neutral_pca_status": "pending B7-B8; these are raw directions"}
    (output / "vectors_metadata.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Saved raw class means and directions: {output}", flush=True)


def run_model(key, data, source_hash, b2_hash, args):
    model_id = MODELS[key]
    print(f"Loading {model_id} on CUDA", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16,
                                                 local_files_only=True).eval().to("cuda")
    wrapped = NNsight(base)
    layers = len(base.model.layers)
    width = base.config.hidden_size
    signature = {"model_id": model_id, "source_sha256": source_hash,
                 "b2_vector_train_sha256": b2_hash,
                 "layers": layers, "hidden_size": width, "examples": len(data),
                 "text_token_rule": "raw text; add_special_tokens=False; exclude padding and special IDs",
                 "pooling": "mean over valid text-token residual outputs after each decoder block",
                 "max_input_tokens": MAX_TOKENS, "activation_storage_dtype": "float16",
                 "model_forward_dtype": "float16", "nnsight_version": nnsight.__version__,
                 "torch_version": torch.__version__, "transformers_version": transformers.__version__}
    output = args.output / key
    conn, pooled = checkpoint(output, signature, len(data), layers, width)
    try:
        completed = {row[0] for row in conn.execute("SELECT row_index FROM completed")}
        selected = range(min(args.limit, len(data))) if args.limit else range(len(data))
        pending = [index for index in selected if index not in completed]
        print(f"{key}: {len(completed):,} saved, {len(pending):,} pending; "
              f"{layers} layers x {width} hidden", flush=True)
        for start in range(0, len(pending), args.batch_size):
            indices = pending[start:start + args.batch_size]
            batch = [data[index] for index in indices]
            encoded, mask, lengths = tokenized_batch(tokenizer, batch)
            values = extract_pooled(wrapped, encoded, mask, layers)
            pooled[indices] = values
            pooled.flush()  # Persist vectors before marking rows complete.
            records = [(index, row["emotion"], row["id"], token_count,
                        hashlib.sha256(row["text"].encode("utf-8")).hexdigest())
                       for index, row, token_count in zip(indices, batch, lengths)]
            with conn:
                conn.executemany("INSERT INTO completed VALUES (?,?,?,?,?)", records)
            if start % (args.batch_size * 25) == 0 or start + len(indices) == len(pending):
                print(f"  {key}: {len(completed) + start + len(indices):,}/{len(data):,} complete", flush=True)
        total = conn.execute("SELECT COUNT(*) FROM completed").fetchone()[0]
        if total == len(data):
            export_vectors(data, pooled, output, model_id, signature)
        else:
            print(f"{key}: partial checkpoint {total:,}/{len(data):,}; resume without --limit", flush=True)
    finally:
        conn.close()
        del wrapped, base, tokenizer, pooled
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODELS, "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Only first N rows for a smoke run; resume without it")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "stage_b_raw_vectors")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch-size and limit must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    print(f"CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    data, source_hash, b2_hash = load_training()
    args.output = args.output.resolve()
    for key in MODELS if args.model == "all" else (args.model,):
        run_model(key, data, source_hash, b2_hash, args)


if __name__ == "__main__":
    main()
