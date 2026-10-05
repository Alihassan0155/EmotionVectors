"""Audit geometric changes from Stage B neutral-PCA denoising."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = (
    ("0.5b", "Qwen 0.5B"),
    ("1.5b", "Qwen 1.5B"),
    ("3b", "Qwen 3B"),
    ("bert_large_uncased", "BERT-large"),
)


def load_model(root: Path, key: str):
    raw_folder = root / "artifacts" / "stage_b_raw_vectors" / key
    final_folder = root / "artifacts" / "stage_b_final_vectors" / key
    raw_path = raw_folder / "emotion_directions_raw.float32.npy"
    final_path = final_folder / "emotion_directions_final.float32.npy"
    raw_meta = json.loads((raw_folder / "vectors_metadata.json").read_text(encoding="utf-8"))
    final_meta = json.loads((final_folder / "final_vectors_metadata.json").read_text(encoding="utf-8"))
    if not raw_meta.get("complete") or not final_meta.get("complete"):
        raise ValueError(f"Incomplete Stage B vectors: {key}")
    if raw_meta["model_id"] != final_meta["model_id"] or raw_meta["emotion_order"] != final_meta["emotion_order"]:
        raise ValueError(f"Raw and final metadata disagree: {key}")
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if raw_hash != final_meta["raw_directions_sha256"]:
        raise ValueError(f"Final vectors were computed from a different raw file: {key}")
    raw = np.load(raw_path).astype(np.float64)
    final = np.load(final_path).astype(np.float64)
    expected = (8, raw_meta["layers"], raw_meta["hidden_size"])
    if raw.shape != expected or final.shape != expected:
        raise ValueError(f"Unexpected vector shape: {key}")
    if not np.isfinite(raw).all() or not np.isfinite(final).all():
        raise ValueError(f"Non-finite vectors: {key}")
    return raw, final, raw_meta["emotion_order"]


def measures(raw, final):
    raw_norm = np.linalg.norm(raw, axis=2)
    final_norm = np.linalg.norm(final, axis=2)
    if np.any(raw_norm <= 0) or np.any(final_norm <= 0):
        raise ValueError("Zero-length raw or final direction")
    ratio = final_norm / raw_norm
    cosine = np.einsum("eld,eld->el", raw, final) / (raw_norm * final_norm)
    cosine = np.clip(cosine, -1, 1)
    angle = np.degrees(np.arccos(cosine))
    identity_error = np.abs(cosine - ratio)
    if np.max(identity_error) > 1e-3:
        raise ValueError("Projection identity failed; inspect final vectors")
    return cosine, ratio, angle, identity_error


def save_plot(output, key, label, emotions, cosine, ratio, angle):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.titlesize": 14, "figure.titleweight": "bold",
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(3, 1, figsize=(max(10, cosine.shape[1] * .36), 8.5),
                             constrained_layout=True)
    settings = (
        (cosine, "Cosine(raw, final)", "viridis", 0, 1),
        (ratio, "Final norm / raw norm", "viridis", 0, 1),
        (angle, "Angle between directions (degrees)", "magma", 0, 90),
    )
    for ax, (values, title, cmap, vmin, vmax) in zip(axes, settings):
        image = ax.imshow(values, aspect="auto", interpolation="nearest",
                          cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set(yticks=np.arange(8), ylabel="Emotion", title=title)
        ax.set_yticklabels(emotions)
        ax.set_xticks(np.arange(cosine.shape[1]))
        ax.set_xticklabels(np.arange(1, cosine.shape[1] + 1))
        ax.set_xlabel("Layer (1 based)")
        ax.tick_params(axis="x", labelsize=8)
        fig.colorbar(image, ax=ax, fraction=.022, pad=.012)
    fig.suptitle(f"{label}: raw to neutral-PCA-denoised emotion directions")
    stem = f"stage_b_denoising_geometry_{key}"
    fig.savefig(output / f"{stem}.png", dpi=350, bbox_inches="tight", pad_inches=.15)
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight", pad_inches=.15)
    plt.close(fig)


def write_outputs(root, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = ["STAGE B - RAW VS NEUTRAL-PCA-DENOISED VECTOR GEOMETRY", "",
               "Every metric uses matched emotion and layer vectors.",
               "For orthogonal projection, cosine(raw, final) = ||final|| / ||raw||.",
               "Angle = arccos(cosine), in degrees. The three metrics describe the same projection.",
               "They do not measure emotion specificity or downstream usefulness.", ""]
    for key, label in MODELS:
        raw, final, emotions = load_model(root, key)
        cosine, ratio, angle, error = measures(raw, final)
        save_plot(output, key, label, emotions, cosine, ratio, angle)
        for emotion_index, emotion in enumerate(emotions):
            for layer_index in range(raw.shape[1]):
                rows.append((key, label, emotion, layer_index + 1,
                             float(cosine[emotion_index, layer_index]),
                             float(ratio[emotion_index, layer_index]),
                             float(angle[emotion_index, layer_index])))
        summary += [label,
                    f"  Vectors: {cosine.size} (8 emotions x {raw.shape[1]} layers)",
                    f"  Cosine, median [min, max]: {np.median(cosine):.3f} "
                    f"[{cosine.min():.3f}, {cosine.max():.3f}]",
                    f"  Norm ratio, median [min, max]: {np.median(ratio):.3f} "
                    f"[{ratio.min():.3f}, {ratio.max():.3f}]",
                    f"  Angle, median [min, max]: {np.median(angle):.1f} deg "
                    f"[{angle.min():.1f}, {angle.max():.1f}]",
                    f"  Max |cosine - norm ratio|: {error.max():.2e}", ""]
    csv_path = output / "stage_b_denoising_geometry.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model_key", "model", "emotion", "layer_1_based",
                         "cosine_raw_final", "final_to_raw_norm_ratio", "angle_degrees"))
        writer.writerows(rows)
    summary += [f"Per-emotion, per-layer results: {csv_path.name}",
                "Figures: stage_b_denoising_geometry_<model>.png and .svg"]
    summary_path = output / "stage_b_denoising_geometry_summary.txt"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(summary_path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    write_outputs(root, (args.output or root / "visualizations").resolve())


if __name__ == "__main__":
    main()
