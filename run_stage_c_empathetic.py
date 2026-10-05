"""Stage C6-C7: external EmpatheticDialogues AUROC and intervention layers.

Runs the frozen 490-conversation sample through the same Stage B/C extraction
path, scores existing final vectors, compares matched seven-class GoEmotions
AUROC, and selects a layer using only these emotion-validation results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

from build_stage_b_final_vectors import MODEL_IDS, atomic_json, atomic_npy
from prepare_empathetic_stage_c import LABEL_MAP
from run_stage_c_goemotions import auroc, extract, vector_inputs


EMOTIONS = tuple(LABEL_MAP)
WINDOW = 3


def frozen_sample(folder):
    sample_path = folder / "sample.jsonl"
    manifest = json.loads((folder / "sample_manifest.json").read_text(encoding="utf-8"))
    sample_bytes = sample_path.read_bytes()
    if hashlib.sha256(sample_bytes).hexdigest() != manifest["sample_sha256"]:
        raise ValueError("Frozen EmpatheticDialogues sample hash mismatch")
    rows = [json.loads(line) for line in sample_bytes.decode("utf-8").splitlines()]
    if (len(rows) != 490 or manifest["sample_size"] != 490 or
            set(row["emotion"] for row in rows) != set(EMOTIONS) or
            len({row["id"] for row in rows}) != 490 or
            any(row["source_label"] != LABEL_MAP[row["emotion"]] for row in rows)):
        raise ValueError("Frozen EmpatheticDialogues sample is invalid")
    return rows, manifest


def ed_signature(key, sample_manifest, raw_meta):
    return {
        "model_id": MODEL_IDS[key], "source": "EmpatheticDialogues official valid.csv situation prompts",
        "sample_sha256": sample_manifest["sample_sha256"],
        "source_archive_sha256": sample_manifest["source_archive_sha256"],
        "layers": raw_meta["layers"], "hidden_size": raw_meta["hidden_size"],
        "examples": sample_manifest["sample_size"],
        "pooling": raw_meta["pooling"], "text_token_rule": raw_meta["text_token_rule"],
        "activation_site": raw_meta.get("activation_site", "decoder block output"),
        "activation_storage_dtype": "float16", "model_forward_dtype": "float16",
        "label_map": LABEL_MAP,
    }


def matched_goemotions(key, layers):
    folder = ROOT / "artifacts" / "stage_c_goemotions" / key
    scores = np.load(folder / "scores_final.float32.npy")
    with (folder / "validation_examples.csv").open(newline="", encoding="utf-8") as handle:
        labels = [row["emotion"] for row in csv.DictReader(handle)]
    all_order = ("joy", "love", "excitement", "gratitude", "sadness", "anger", "fear", "disgust")
    if scores.shape != (len(labels), layers, len(all_order)):
        raise ValueError(f"GoEmotions score shape mismatch: {key}")
    keep = np.array([label in EMOTIONS for label in labels])
    labels = np.asarray(labels)[keep]
    scores = scores[keep]
    matrix = np.empty((len(EMOTIONS), layers), dtype=np.float64)
    for i, emotion in enumerate(EMOTIONS):
        column = all_order.index(emotion)
        positives = labels == emotion
        for layer in range(layers):
            matrix[i, layer] = auroc(positives, scores[:, layer, column])
    return matrix, len(labels)


def evaluate(key, rows, signature, folder, final):
    pooled = np.load(folder / "pooled.float16.npy", mmap_mode="r")
    n, layers, width = pooled.shape
    if (n, layers, width) != (len(rows), signature["layers"], signature["hidden_size"]):
        raise ValueError("External activation shape mismatch")
    all_order = ("joy", "love", "excitement", "gratitude", "sadness", "anger", "fear", "disgust")
    chosen = np.stack([final[all_order.index(emotion)] for emotion in EMOTIONS]).astype(np.float64)
    norms = np.linalg.norm(chosen, axis=2, keepdims=True)
    if not np.isfinite(chosen).all() or np.any(norms <= 0):
        raise ValueError("Invalid final emotion vectors")
    chosen /= norms
    scores = np.empty((n, layers, len(EMOTIONS)), dtype=np.float32)
    for layer in range(layers):
        h = np.asarray(pooled[:, layer, :], dtype=np.float64)
        scores[:, layer] = h @ chosen[:, layer, :].T
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite external projection scores")
    atomic_npy(folder / "scores_final.float32.npy", scores)
    labels = np.array([row["emotion"] for row in rows])
    ed = np.empty((len(EMOTIONS), layers), dtype=np.float64)
    for i, emotion in enumerate(EMOTIONS):
        positives = labels == emotion
        for layer in range(layers):
            ed[i, layer] = auroc(positives, scores[:, layer, i])
    go, go_count = matched_goemotions(key, layers)
    atomic_npy(folder / "auroc_empathetic_final.float64.npy", ed)
    atomic_npy(folder / "auroc_goemotions_matched7.float64.npy", go)
    atomic_npy(folder / "auroc_drop_go_minus_empathetic.float64.npy", go - ed)
    with (folder / "auroc_by_emotion_layer.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "emotion", "ed_source_label", "layer_1_based",
                         "empathetic_auroc", "goemotions_matched7_auroc", "drop_go_minus_empathetic"))
        for i, emotion in enumerate(EMOTIONS):
            for layer in range(layers):
                writer.writerow((MODEL_IDS[key], emotion, LABEL_MAP[emotion], layer + 1,
                                 ed[i, layer], go[i, layer], go[i, layer] - ed[i, layer]))
    with (folder / "sample_examples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_index", "conversation_id", "emotion", "source_label", "text_sha256"))
        for i, row in enumerate(rows):
            writer.writerow((i, row["id"], row["emotion"], row["source_label"],
                             hashlib.sha256(row["text"].encode("utf-8")).hexdigest()))
    result = {**signature, "complete": True,
              "score_definition": "dot(pooled activation, unit-normalized existing final direction)",
              "auroc_definition": "one emotion vs the other six, per layer",
              "goemotions_comparison": "recomputed from saved C1-C5 scores after excluding love examples",
              "goemotions_matched7_examples": go_count,
              "empathetic_examples_per_emotion": 70,
              "macro_auroc_empathetic_by_layer": ed.mean(axis=0).tolist(),
              "macro_auroc_goemotions_matched7_by_layer": go.mean(axis=0).tolist(),
              "macro_drop_go_minus_empathetic_by_layer": (go - ed).mean(axis=0).tolist(),
              "mean_auroc_empathetic_all_cells": float(ed.mean()),
              "mean_auroc_goemotions_matched7_all_cells": float(go.mean())}
    atomic_json(folder / "results_metadata.json", result)
    print(f"{key}: mean matched GoEmotions AUROC {go.mean():.3f}; "
          f"EmpatheticDialogues {ed.mean():.3f}", flush=True)
    return result


def select_layer(key, folder):
    go = np.load(folder / "auroc_goemotions_matched7.float64.npy")
    ed = np.load(folder / "auroc_empathetic_final.float64.npy")
    if go.shape != ed.shape or go.shape[0] != len(EMOTIONS):
        raise ValueError("Cannot select from mismatched AUROC arrays")
    candidates = []
    for center in range(1, go.shape[1] - 1):
        go_window = go[:, center - 1:center + 2]
        ed_window = ed[:, center - 1:center + 2]
        go_macro = float(go_window.mean())
        ed_macro = float(ed_window.mean())
        go_q25 = float(np.quantile(go_window.mean(axis=1), .25))
        ed_q25 = float(np.quantile(ed_window.mean(axis=1), .25))
        score = .7 * min(go_macro, ed_macro) + .3 * min(go_q25, ed_q25)
        candidates.append({"layer_1_based": center + 1,
                           "normalized_depth": center / (go.shape[1] - 1),
                           "selection_score": score,
                           "go_window_macro_auroc": go_macro,
                           "empathetic_window_macro_auroc": ed_macro,
                           "go_window_emotion_q25": go_q25,
                           "empathetic_window_emotion_q25": ed_q25,
                           "go_at_selected_layer_macro_auroc": float(go[:, center].mean()),
                           "empathetic_at_selected_layer_macro_auroc": float(ed[:, center].mean())})
    selected = max(candidates, key=lambda x: (x["selection_score"], -x["layer_1_based"]))
    return selected, candidates


def summarize(output_root, results):
    selection = {
        "rule": "For each 3-layer centered window, score = 0.7*min(Go macro, ED macro) + 0.3*min(Go emotion q25, ED emotion q25); maximize, tie earliest layer.",
        "emotion_order": EMOTIONS,
        "uses_only": ["GoEmotions held-out emotion validation", "EmpatheticDialogues validation sample"],
        "does_not_use": ["SST-2", "TweetEval"],
        "caution": "EmpatheticDialogues also influences layer choice, so selected-layer external AUROC is descriptive; reserve an untouched test set for unbiased final evaluation.",
        "models": {},
    }
    lines = ["STAGE C6-C7 - EXTERNAL EMOTION VALIDATION AND LAYER SELECTION", "",
             "External data: 490 unique EmpatheticDialogues validation situations; 70 per emotion.",
             "Seven direct mappings only; love excluded because the dataset has no exact love label.",
             "Vectors: existing neutral-PCA-denoised GoEmotions directions; no refitting.",
             "Scores: pooled activation dot unit-normalized direction; AUROC vs other six.",
             "GoEmotions comparison recomputes AUROC after excluding love examples.",
             "Selection uses both emotion-validation datasets, never SST-2 or TweetEval.",
             "External AUROC at the selected layer is descriptive because ED informed selection.", ""]
    for key in MODEL_IDS:
        if key not in results:
            continue
        folder = output_root / key
        chosen, candidates = select_layer(key, folder)
        selection["models"][key] = {"model_id": MODEL_IDS[key], "selected": chosen,
                                    "all_candidate_windows": candidates}
        result = results[key]
        go = result["mean_auroc_goemotions_matched7_all_cells"]
        ed = result["mean_auroc_empathetic_all_cells"]
        lines += [MODEL_IDS[key],
                  f"  Mean AUROC across layers: GoEmotions {go:.3f}; EmpatheticDialogues {ed:.3f}; drop {go-ed:+.3f}",
                  f"  Selected layer: {chosen['layer_1_based']} / {result['layers']} "
                  f"(normalized depth {chosen['normalized_depth']:.3f})",
                  f"  At selected layer: Go {chosen['go_at_selected_layer_macro_auroc']:.3f}; "
                  f"ED {chosen['empathetic_at_selected_layer_macro_auroc']:.3f}",
                  f"  Selection score: {chosen['selection_score']:.3f}", ""]
    atomic_json(output_root / "selected_intervention_layers.json", selection)
    (output_root / "stage_c6_c7_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((output_root / "stage_c6_c7_results.txt").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODEL_IDS, "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Checkpoint smoke test on first N rows")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Batch size and limit must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    print(f"CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    folder = ROOT / "artifacts" / "stage_c_empathetic_dialogues"
    rows, sample_manifest = frozen_sample(folder)
    b2_manifest = json.loads((ROOT / "artifacts" / "goemotions_b2" / "manifest.json").read_text(encoding="utf-8"))
    results = {}
    keys = tuple(MODEL_IDS) if args.model == "all" else (args.model,)
    for key in keys:
        raw_meta, _, final = vector_inputs(key, b2_manifest)
        signature = ed_signature(key, sample_manifest, raw_meta)
        model_folder = folder / key
        complete = extract(key, rows, signature, model_folder, args.batch_size, args.limit)
        if complete and args.limit is None:
            results[key] = evaluate(key, rows, signature, model_folder, final)
        else:
            print(f"{key}: partial sample checkpoint; resume without --limit", flush=True)
    for key in MODEL_IDS:
        path = folder / key / "results_metadata.json"
        if key not in results and path.exists():
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored.get("complete"):
                results[key] = stored
    if results:
        summarize(folder, results)


if __name__ == "__main__":
    main()
