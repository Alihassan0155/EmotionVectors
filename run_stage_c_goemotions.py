"""Stage C1-C5: held-out GoEmotions extraction and raw/final AUROC.

Only the frozen B2 emotion_validation split is scored. Scores are unit-vector
projections of pooled activations; one-vs-rest AUROC is computed per layer.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
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
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from build_stage_b_vectors import (EMOTIONS, checkpoint, dataset_digest,
                                   extract_pooled as extract_qwen,
                                   tokenized_batch as tokenize_qwen)
from build_stage_b_bert_vectors import (extract_pooled as extract_bert,
                                        tokenized_batch as tokenize_bert)
from build_stage_b_final_vectors import MODEL_IDS, atomic_json, atomic_npy
from prepare_goemotions_b2 import split_digest


def validation_data():
    folder = ROOT / "artifacts" / "goemotions_b2"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    data = load_from_disk(str(folder / "emotion_validation"))
    if (len(data) != manifest["counts"]["emotion_validation"] or
            split_digest(data) != manifest["sha256"]["emotion_validation"]):
        raise ValueError("Held-out emotion validation split differs from B2 manifest")
    if any(row["source_split"] != "validation" or row["emotion"] not in EMOTIONS or
           len(row["labels"]) != 1 for row in data):
        raise ValueError("Validation split contains an unexpected row")
    return data, dataset_digest(data), manifest


def vector_inputs(key, manifest):
    raw_folder = ROOT / "artifacts" / "stage_b_raw_vectors" / key
    final_folder = ROOT / "artifacts" / "stage_b_final_vectors" / key
    raw_meta = json.loads((raw_folder / "vectors_metadata.json").read_text(encoding="utf-8"))
    final_meta = json.loads((final_folder / "final_vectors_metadata.json").read_text(encoding="utf-8"))
    if not raw_meta.get("complete") or not final_meta.get("complete"):
        raise ValueError(f"Incomplete Stage B vectors: {key}")
    if raw_meta["model_id"] != MODEL_IDS[key] or final_meta["model_id"] != MODEL_IDS[key]:
        raise ValueError(f"Model identifier mismatch: {key}")
    if raw_meta["b2_vector_train_sha256"] != manifest["sha256"]["vector_train"]:
        raise ValueError(f"Wrong emotion-vector train split: {key}")
    if final_meta["neutral_signature"]["b2_pca_neutral_train_sha256"] != manifest["sha256"]["pca_neutral_train"]:
        raise ValueError(f"Wrong neutral PCA train split: {key}")
    if list(EMOTIONS) != raw_meta["emotion_order"] or list(EMOTIONS) != final_meta["emotion_order"]:
        raise ValueError(f"Emotion order mismatch: {key}")
    raw_file = raw_folder / "emotion_directions_raw.float32.npy"
    final_file = final_folder / "emotion_directions_final.float32.npy"
    if hashlib.sha256(raw_file.read_bytes()).hexdigest() != final_meta["raw_directions_sha256"]:
        raise ValueError(f"Final vectors do not match raw source: {key}")
    raw = np.load(raw_file)
    final = np.load(final_file)
    expected = (len(EMOTIONS), raw_meta["layers"], raw_meta["hidden_size"])
    if raw.shape != expected or final.shape != expected:
        raise ValueError(f"Vector dimensions mismatch: {key}")
    return raw_meta, raw, final


def signature_for(key, data_hash, manifest, raw_meta, examples):
    return {
        "model_id": MODEL_IDS[key], "source": "goemotions_b2/emotion_validation",
        "source_sha256": data_hash,
        "b2_emotion_validation_sha256": manifest["sha256"]["emotion_validation"],
        "layers": raw_meta["layers"], "hidden_size": raw_meta["hidden_size"],
        "examples": examples, "pooling": raw_meta["pooling"],
        "text_token_rule": raw_meta["text_token_rule"],
        "activation_site": raw_meta.get("activation_site", "decoder block output"),
        "activation_storage_dtype": "float16", "model_forward_dtype": "float16",
        "nnsight_version": nnsight.__version__, "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }


def extract(key, data, signature, folder, batch_size, limit):
    layers, width = signature["layers"], signature["hidden_size"]
    conn, pooled = checkpoint(folder, signature, len(data), layers, width)
    try:
        completed = {row[0] for row in conn.execute("SELECT row_index FROM completed")}
        selected = range(min(limit, len(data))) if limit else range(len(data))
        pending = [i for i in selected if i not in completed]
        print(f"{key}: {len(completed)} validation saved, {len(pending)} pending", flush=True)
        if pending:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS[key], local_files_only=True)
            if key == "bert_large_uncased":
                base = AutoModel.from_pretrained(MODEL_IDS[key], dtype=torch.float16,
                                                  local_files_only=True).eval().to("cuda")
            else:
                tokenizer.padding_side = "right"
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                base = AutoModelForCausalLM.from_pretrained(
                    MODEL_IDS[key], dtype=torch.float16, local_files_only=True).eval().to("cuda")
            wrapped = NNsight(base)
            actual_layers = len(base.encoder.layer) if key == "bert_large_uncased" else len(base.model.layers)
            if actual_layers != layers or base.config.hidden_size != width:
                raise ValueError("Model differs from Stage B vector dimensions")
            try:
                for start in range(0, len(pending), batch_size):
                    indices = pending[start:start + batch_size]
                    batch = [data[i] for i in indices]
                    if key == "bert_large_uncased":
                        inputs, mask, lengths = tokenize_bert(tokenizer, batch)
                        values = extract_bert(wrapped, inputs, mask, layers)
                    else:
                        inputs, mask, lengths = tokenize_qwen(tokenizer, batch)
                        values = extract_qwen(wrapped, inputs, mask, layers)
                    pooled[indices] = values
                    pooled.flush()
                    records = [(i, row["emotion"], row["id"], length,
                                hashlib.sha256(row["text"].encode("utf-8")).hexdigest())
                               for i, row, length in zip(indices, batch, lengths)]
                    with conn:
                        conn.executemany("INSERT INTO completed VALUES (?,?,?,?,?)", records)
                    if start % (batch_size * 25) == 0 or start + len(indices) == len(pending):
                        print(f"  {key}: {len(completed) + start + len(indices)}/{len(data)} validation", flush=True)
            finally:
                del wrapped, base, tokenizer
                gc.collect()
                torch.cuda.empty_cache()
        total = conn.execute("SELECT COUNT(*) FROM completed").fetchone()[0]
        return total == len(data)
    finally:
        conn.close()
        del pooled


def auroc(y, scores):
    """Mann-Whitney AUROC with average ranks for exact score ties."""
    y = np.asarray(y, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives or not np.isfinite(scores).all():
        raise ValueError("AUROC requires both classes and finite scores")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[y].sum() - positives * (positives + 1) / 2) /
                 (positives * negatives))


def scores_and_metrics(key, data, signature, folder, raw, final):
    pooled = np.load(folder / "pooled.float16.npy", mmap_mode="r")
    n, layers, width = pooled.shape
    if (n, layers, width) != (len(data), signature["layers"], signature["hidden_size"]):
        raise ValueError("Validation activation shape mismatch")
    labels = np.array([EMOTIONS.index(row["emotion"]) for row in data])
    matrices = {}
    for kind, directions in (("raw", raw), ("final", final)):
        unit = directions.astype(np.float64)
        norm = np.linalg.norm(unit, axis=2, keepdims=True)
        if np.any(norm <= 0) or not np.isfinite(norm).all():
            raise ValueError(f"Invalid {kind} direction norms")
        unit /= norm
        scores = np.empty((n, layers, len(EMOTIONS)), dtype=np.float32)
        for layer in range(layers):
            h = np.asarray(pooled[:, layer, :], dtype=np.float64)
            scores[:, layer, :] = h @ unit[:, layer, :].T
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite {kind} validation scores")
        atomic_npy(folder / f"scores_{kind}.float32.npy", scores)
        matrix = np.empty((len(EMOTIONS), layers), dtype=np.float64)
        for emotion_index in range(len(EMOTIONS)):
            is_positive = labels == emotion_index
            for layer in range(layers):
                matrix[emotion_index, layer] = auroc(is_positive, scores[:, layer, emotion_index])
        matrices[kind] = matrix
        atomic_npy(folder / f"auroc_{kind}.float64.npy", matrix)
    atomic_npy(folder / "auroc_final_minus_raw.float64.npy", matrices["final"] - matrices["raw"])
    with (folder / "auroc_by_emotion_layer.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "emotion", "positive_examples", "negative_examples",
                         "layer_1_based", "auroc_raw", "auroc_final", "delta_final_minus_raw"))
        for i, emotion in enumerate(EMOTIONS):
            pos = int((labels == i).sum())
            for layer in range(layers):
                writer.writerow((MODEL_IDS[key], emotion, pos, n - pos, layer + 1,
                                 matrices["raw"][i, layer], matrices["final"][i, layer],
                                 matrices["final"][i, layer] - matrices["raw"][i, layer]))
    with (folder / "validation_examples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_index", "id", "emotion", "text_sha256"))
        for index, row in enumerate(data):
            writer.writerow((index, row["id"], row["emotion"],
                             hashlib.sha256(row["text"].encode("utf-8")).hexdigest()))
    result = {**signature, "complete": True, "score_definition":
              "dot(pooled activation, unit-normalized emotion direction)",
              "auroc_definition": "emotion vs the other seven emotions; average ranks for ties",
              "class_counts": {emotion: int((labels == i).sum())
                               for i, emotion in enumerate(EMOTIONS)},
              "macro_auroc_by_layer_raw": matrices["raw"].mean(axis=0).tolist(),
              "macro_auroc_by_layer_final": matrices["final"].mean(axis=0).tolist(),
              "mean_auroc_raw_all_cells": float(matrices["raw"].mean()),
              "mean_auroc_final_all_cells": float(matrices["final"].mean()),
              "mean_delta_final_minus_raw_all_cells": float((matrices["final"] - matrices["raw"]).mean()),
              "stage_c_layer_selection": "pending separate preregistered selection rule"}
    atomic_json(folder / "results_metadata.json", result)
    print(f"{key}: mean AUROC raw {result['mean_auroc_raw_all_cells']:.3f}, "
          f"final {result['mean_auroc_final_all_cells']:.3f}", flush=True)
    return result


def write_summary(results, root):
    lines = ["STAGE C1-C5 - HELD-OUT GOEMOTIONS VALIDATION", "",
             "879 official validation examples; strict single-label, 8 emotions.",
             "Scores are projections onto unit raw/final directions at each layer.",
             "AUROC compares each emotion to the other seven, with ties handled by average ranks.",
             "No sentiment dataset or GoEmotions test examples used.",
             "AUROC below 0.5 means the chosen direction is inversely associated with its label.",
             "Layer selection and cross-dataset evaluation are separate steps.", ""]
    for key in MODEL_IDS:
        if key not in results:
            continue
        result = results[key]
        raw = np.asarray(result["macro_auroc_by_layer_raw"])
        final = np.asarray(result["macro_auroc_by_layer_final"])
        lines += [MODEL_IDS[key],
                  f"  Mean raw AUROC across 8 emotions x {len(raw)} layers: {result['mean_auroc_raw_all_cells']:.3f}",
                  f"  Mean final AUROC across 8 emotions x {len(final)} layers: {result['mean_auroc_final_all_cells']:.3f}",
                  f"  Mean final minus raw: {result['mean_delta_final_minus_raw_all_cells']:+.3f}",
                  f"  Peak macro AUROC (descriptive only): raw {raw.max():.3f} at layer {raw.argmax()+1}; "
                  f"final {final.max():.3f} at layer {final.argmax()+1}", ""]
    lines += ["Full per-emotion, per-layer results: <model>/auroc_by_emotion_layer.csv",
              "Per-example scores: <model>/scores_raw.float32.npy and scores_final.float32.npy"]
    (root / "stage_c_goemotions_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODEL_IDS, "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Extract only first N rows to test checkpointing")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Batch size and limit must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for activation extraction")
    print(f"CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    data, data_hash, manifest = validation_data()
    output_root = ROOT / "artifacts" / "stage_c_goemotions"
    results = {}
    keys = tuple(MODEL_IDS) if args.model == "all" else (args.model,)
    for key in keys:
        raw_meta, raw, final = vector_inputs(key, manifest)
        signature = signature_for(key, data_hash, manifest, raw_meta, len(data))
        folder = output_root / key
        complete = extract(key, data, signature, folder, args.batch_size, args.limit)
        if complete and args.limit is None:
            results[key] = scores_and_metrics(key, data, signature, folder, raw, final)
        else:
            print(f"{key}: partial checkpoint; resume without --limit", flush=True)
    for key in MODEL_IDS:
        path = output_root / key / "results_metadata.json"
        if key not in results and path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("complete"):
                results[key] = stored
    if results:
        write_summary(results, output_root)
        print(f"Summary: {output_root / 'stage_c_goemotions_results.txt'}", flush=True)


if __name__ == "__main__":
    main()
