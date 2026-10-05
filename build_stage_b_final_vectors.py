"""B7-B9: neutral activations, exact 50%-variance PCA, final emotion vectors.

Uses GoEmotions pca_neutral_train only for PCA fitting. Raw B3-B6 vectors are
read-only inputs; Stage C validation and sentiment datasets are never loaded.
"""

from __future__ import annotations

import argparse
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

from build_stage_b_vectors import (EMOTIONS, MODELS, checkpoint, dataset_digest,
                                   extract_pooled as extract_qwen,
                                   tokenized_batch as tokenize_qwen)
from build_stage_b_bert_vectors import (MODEL_ID as BERT_ID,
                                        extract_pooled as extract_bert,
                                        tokenized_batch as tokenize_bert)
from prepare_goemotions_b2 import split_digest


MODEL_IDS = {**MODELS, "bert_large_uncased": BERT_ID}
PCA_THRESHOLD = 0.5


def neutral_training():
    folder = ROOT / "artifacts" / "goemotions_b2"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    data = load_from_disk(str(folder / "pca_neutral_train"))
    if (len(data) != manifest["counts"]["pca_neutral_train"] or
            split_digest(data) != manifest["sha256"]["pca_neutral_train"]):
        raise ValueError("Neutral training pool differs from B2 manifest")
    if any(row["emotion"] != "neutral" or row["source_split"] != "train" or
           len(row["labels"]) != 1 for row in data):
        raise ValueError("Neutral PCA pool contains a non-neutral or non-train row")
    return data, dataset_digest(data), manifest


def model_dimensions(key):
    folder = ROOT / "artifacts" / "stage_b_raw_vectors" / key
    raw_meta = json.loads((folder / "vectors_metadata.json").read_text(encoding="utf-8"))
    if not raw_meta.get("complete") or raw_meta["model_id"] != MODEL_IDS[key]:
        raise ValueError(f"Incomplete raw vectors for {key}")
    if tuple(raw_meta["emotion_order"]) != EMOTIONS:
        raise ValueError(f"Unexpected emotion order for {key}")
    return raw_meta


def neutral_signature(key, data_hash, manifest, raw_meta, examples):
    return {
        "model_id": MODEL_IDS[key], "neutral_source_sha256": data_hash,
        "b2_pca_neutral_train_sha256": manifest["sha256"]["pca_neutral_train"],
        "b2_vector_train_sha256": manifest["sha256"]["vector_train"],
        "layers": raw_meta["layers"], "hidden_size": raw_meta["hidden_size"],
        "examples": examples, "pooling": raw_meta["pooling"],
        "text_token_rule": raw_meta["text_token_rule"],
        "activation_site": raw_meta.get("activation_site", "decoder block output"),
        "activation_storage_dtype": "float16", "model_forward_dtype": "float16",
        "nnsight_version": nnsight.__version__, "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }


def extract_neutral(key, data, signature, folder, batch_size, limit):
    layers, width = signature["layers"], signature["hidden_size"]
    conn, pooled = checkpoint(folder, signature, len(data), layers, width)
    try:
        completed = {row[0] for row in conn.execute("SELECT row_index FROM completed")}
        selected = range(min(limit, len(data))) if limit else range(len(data))
        pending = [i for i in selected if i not in completed]
        print(f"{key}: {len(completed):,} neutral saved, {len(pending):,} pending", flush=True)
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
                raise ValueError("Raw vector dimensions differ from loaded model")
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
                    if not np.isfinite(values).all():
                        raise ValueError("Non-finite pooled neutral activations")
                    pooled[indices] = values
                    pooled.flush()
                    records = [(i, "neutral", row["id"], length,
                                hashlib.sha256(row["text"].encode("utf-8")).hexdigest())
                               for i, row, length in zip(indices, batch, lengths)]
                    with conn:
                        conn.executemany("INSERT INTO completed VALUES (?,?,?,?,?)", records)
                    if start % (batch_size * 100) == 0 or start + len(indices) == len(pending):
                        print(f"  {key}: {len(completed) + start + len(indices):,}/{len(data):,} neutral", flush=True)
            finally:
                del wrapped, base, tokenizer
                gc.collect()
                torch.cuda.empty_cache()
        total = conn.execute("SELECT COUNT(*) FROM completed").fetchone()[0]
        if total == len(data):
            print(f"{key}: neutral activation matrix complete ({total:,} examples)", flush=True)
        else:
            print(f"{key}: partial neutral checkpoint {total:,}/{len(data):,}", flush=True)
        return total == len(data)
    finally:
        conn.close()
        del pooled


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_npy(path, array):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def pca_remove(neutral_layer, raw_layer):
    """Exact covariance eigendecomposition, then projection of directions.

    Covariance is formed in float32 on CUDA. The neutral mean centers PCA
    fitting; raw contrast directions are projected as displacement vectors.
    """
    x = torch.as_tensor(np.asarray(neutral_layer, dtype=np.float32), device="cuda")
    raw = torch.as_tensor(np.asarray(raw_layer, dtype=np.float32), device="cuda")
    if not torch.isfinite(x).all() or not torch.isfinite(raw).all():
        raise ValueError("Non-finite neutral or raw vectors")
    mean = x.mean(dim=0)
    x = x - mean
    covariance = x.T @ x / (x.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.flip(0).clamp_min(0)
    eigenvectors = eigenvectors.flip(1)
    total_variance = eigenvalues.sum()
    if not torch.isfinite(total_variance) or total_variance <= 0:
        raise ValueError("Neutral variance is non-positive or non-finite")
    cumulative = eigenvalues.cumsum(0) / total_variance
    count = int(torch.searchsorted(cumulative, PCA_THRESHOLD).item()) + 1
    components = eigenvectors[:, :count].T.contiguous()
    final = raw - (raw @ components.T) @ components
    explained = float(cumulative[count - 1].item())
    if explained + 1e-5 < PCA_THRESHOLD or not torch.isfinite(final).all():
        raise ValueError("PCA threshold or final vector check failed")
    removed = 1 - torch.linalg.vector_norm(final, dim=1) / torch.linalg.vector_norm(raw, dim=1)
    result = (mean.cpu().numpy(), components.cpu().numpy(), final.cpu().numpy(),
              {"n_components": count, "explained_variance": explained,
               "total_variance": float(total_variance.item()),
               "mean_norm_fraction_removed": float(removed.mean().item()),
               "min_final_norm": float(torch.linalg.vector_norm(final, dim=1).min().item())})
    del x, raw, covariance, eigenvalues, eigenvectors, components, final
    return result


def denoise(key, signature, neutral_folder, output_folder):
    raw_folder = ROOT / "artifacts" / "stage_b_raw_vectors" / key
    raw = np.load(raw_folder / "emotion_directions_raw.float32.npy", mmap_mode="r")
    neutral = np.load(neutral_folder / "pooled.float16.npy", mmap_mode="r")
    layers, width = signature["layers"], signature["hidden_size"]
    if raw.shape != (8, layers, width) or neutral.shape != (signature["examples"], layers, width):
        raise ValueError("Raw or neutral array dimensions differ from metadata")
    raw_hash = hashlib.sha256((raw_folder / "emotion_directions_raw.float32.npy").read_bytes()).hexdigest()
    run_signature = {"neutral_signature": signature, "raw_directions_sha256": raw_hash,
                     "pca_variance_threshold": PCA_THRESHOLD,
                     "pca_algorithm": "center neutral activations; float32 covariance; CUDA torch.linalg.eigh",
                     "projection": "raw contrast direction minus orthogonal projection onto retained neutral PCs"}
    output_folder.mkdir(parents=True, exist_ok=True)
    signature_file = output_folder / "run_signature.json"
    if signature_file.exists():
        if json.loads(signature_file.read_text(encoding="utf-8")) != run_signature:
            raise ValueError("Existing final-vector output uses different inputs or settings")
    else:
        if any(output_folder.iterdir()):
            raise ValueError("Output folder has files but no run signature")
        atomic_json(signature_file, run_signature)
    components_folder = output_folder / "pca_components"
    components_folder.mkdir(exist_ok=True)
    final_path = output_folder / "emotion_directions_final.float32.npy"
    mean_path = output_folder / "neutral_pca_means.float32.npy"
    if final_path.exists():
        final = np.load(final_path, mmap_mode="r+")
        means = np.load(mean_path, mmap_mode="r+")
        if final.shape != raw.shape or means.shape != (layers, width):
            raise ValueError("Final checkpoint shape differs from raw vectors")
    else:
        final = np.lib.format.open_memmap(final_path, mode="w+", dtype=np.float32,
                                          shape=raw.shape)
        means = np.lib.format.open_memmap(mean_path, mode="w+", dtype=np.float32,
                                          shape=(layers, width))
    stats = []
    for layer in range(layers):
        component_path = components_folder / f"layer_{layer + 1:02d}.float32.npy"
        stats_path = output_folder / f"layer_{layer + 1:02d}_pca.json"
        if stats_path.exists():
            if not component_path.exists():
                raise ValueError(f"Missing PCA components for completed layer {layer + 1}")
            stat = json.loads(stats_path.read_text(encoding="utf-8"))
            if stat["layer_1_based"] != layer + 1:
                raise ValueError("PCA checkpoint layer mismatch")
        else:
            mean, components, denoised, stat = pca_remove(neutral[:, layer, :], raw[:, layer, :])
            means[layer] = mean
            final[:, layer, :] = denoised
            means.flush()
            final.flush()
            atomic_npy(component_path, components)
            stat = {"layer_1_based": layer + 1, **stat}
            atomic_json(stats_path, stat)
            print(f"  {key} layer {layer + 1}/{layers}: {stat['n_components']} PCs, "
                  f"{stat['explained_variance']:.3%} neutral variance", flush=True)
        stats.append(stat)
    if not np.isfinite(final).all() or any(s["explained_variance"] < PCA_THRESHOLD - 1e-5 for s in stats):
        raise ValueError("Final vectors or PCA threshold are invalid")
    result = {
        **run_signature, "complete": True, "model_id": MODEL_IDS[key],
        "emotion_order": EMOTIONS, "array_axes": ["emotion", "layer_1_based", "hidden_dimension"],
        "layers": layers, "hidden_size": width, "neutral_examples": signature["examples"],
        "layer_stats": stats, "stage_c_layer_selection": "pending; use held-out emotion validation only",
    }
    atomic_json(output_folder / "final_vectors_metadata.json", result)
    print(f"{key}: saved final vectors {tuple(final.shape)}", flush=True)
    return result


def write_summary(root, model_results):
    lines = ["STAGE B B7-B9 - NEUTRAL PCA DENOISED EMOTION VECTORS", "",
             "Data: GoEmotions simplified official train split only.",
             "Emotion means: 7,417 strict single-label training examples across 8 emotions.",
             "Neutral PCA: 12,810 separate neutral training examples per model.",
             "At each layer, remove the smallest number of centered neutral PCs explaining >=50% variance.",
             "Project raw contrast directions onto the orthogonal complement of those PCs.",
             "No validation, test, SST-2, or TweetEval examples used here.",
             "No final layer selected; Stage C uses held-out emotion validation.", ""]
    for key, label in ((*[(k, MODEL_IDS[k]) for k in MODELS],
                        ("bert_large_uncased", BERT_ID))):
        if key not in model_results:
            continue
        result = model_results[key]
        stats = result["layer_stats"]
        counts = [row["n_components"] for row in stats]
        variance = [row["explained_variance"] for row in stats]
        removed = [row["mean_norm_fraction_removed"] for row in stats]
        lines += [f"{label}",
                  f"  Final array: {8} emotions x {result['layers']} layers x {result['hidden_size']} hidden",
                  f"  PCs per layer (min / median / max): {min(counts)} / {np.median(counts):g} / {max(counts)}",
                  f"  Neutral variance explained (min / max): {min(variance):.2%} / {max(variance):.2%}",
                  f"  Mean fraction of raw vector norm removed across layers: {np.mean(removed):.2%}",
                  f"  Final directions: artifacts/stage_b_final_vectors/{key}/emotion_directions_final.float32.npy",
                  ""]
    lines += ["Layer-level PCA counts and variance are in each model's final_vectors_metadata.json.",
              "Neutral pooled activation checkpoints are under artifacts/stage_b_neutral_activations/."]
    root.mkdir(parents=True, exist_ok=True)
    (root / "stage_b_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary: {root / 'stage_b_results.txt'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODEL_IDS, "all"), default="all")
    parser.add_argument("--phase", choices=("all", "extract", "denoise"), default="all")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Extract only first N neutral rows for a smoke test")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Batch size and limit must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for neutral extraction and PCA")
    print(f"CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    data, data_hash, manifest = neutral_training()
    keys = tuple(MODEL_IDS) if args.model == "all" else (args.model,)
    output_root = ROOT / "artifacts" / "stage_b_final_vectors"
    model_results = {}
    for key in keys:
        raw_meta = model_dimensions(key)
        if raw_meta["b2_vector_train_sha256"] != manifest["sha256"]["vector_train"]:
            raise ValueError(f"Raw directions for {key} use a different training split")
        signature = neutral_signature(key, data_hash, manifest, raw_meta, len(data))
        neutral_folder = ROOT / "artifacts" / "stage_b_neutral_activations" / key
        complete = True
        if args.phase in ("all", "extract"):
            complete = extract_neutral(key, data, signature, neutral_folder,
                                       args.batch_size, args.limit)
        if args.phase in ("all", "denoise") and complete and not args.limit:
            if args.phase == "denoise":
                import sqlite3
                db = sqlite3.connect(neutral_folder / "progress.sqlite3")
                try:
                    completed = db.execute("SELECT COUNT(*) FROM completed").fetchone()[0]
                    stored_signature = db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
                finally:
                    db.close()
                if completed != len(data) or not stored_signature or json.loads(stored_signature[0]) != signature:
                    raise ValueError(f"Neutral checkpoint for {key} is incomplete or incompatible")
            model_results[key] = denoise(key, signature, neutral_folder, output_root / key)
        elif args.phase in ("all", "denoise"):
            print(f"{key}: final vectors pending complete neutral extraction", flush=True)
    for key in MODEL_IDS:
        metadata_path = output_root / key / "final_vectors_metadata.json"
        if key not in model_results and metadata_path.exists():
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))
            if saved.get("complete"):
                model_results[key] = saved
    if model_results:
        write_summary(output_root, model_results)


if __name__ == "__main__":
    main()
