"""Plot Stage C GoEmotions raw/final one-vs-rest AUROC heatmaps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


MODELS = (("0.5b", "Qwen 0.5B"), ("1.5b", "Qwen 1.5B"),
          ("3b", "Qwen 3B"), ("bert_large_uncased", "BERT-large"))
EMOTIONS = ("joy", "love", "excitement", "gratitude", "sadness",
            "anger", "fear", "disgust")


def plot_model(folder, output, key, label):
    meta = json.loads((folder / "results_metadata.json").read_text(encoding="utf-8"))
    if not meta.get("complete"):
        raise ValueError(f"Incomplete Stage C results: {key}")
    raw = np.load(folder / "auroc_raw.float64.npy")
    final = np.load(folder / "auroc_final.float64.npy")
    delta = final - raw
    if raw.shape != final.shape or raw.shape != (8, meta["layers"]):
        raise ValueError(f"AUROC array shape mismatch: {key}")
    fig, axes = plt.subplots(3, 1, figsize=(max(10, raw.shape[1] * .37), 9),
                             constrained_layout=True)
    for ax, values, title in zip(axes, (raw, final, delta),
                                 ("Raw directions", "Neutral-PCA-denoised directions",
                                  "Denoised minus raw")):
        if title == "Denoised minus raw":
            scale = max(.1, float(np.ceil(np.max(np.abs(delta)) * 10) / 10))
            im = ax.imshow(values, aspect="auto", interpolation="nearest",
                           cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale))
        else:
            im = ax.imshow(values, aspect="auto", interpolation="nearest",
                           cmap="viridis", vmin=0, vmax=1)
        ax.set(title=title, ylabel="Emotion", yticks=np.arange(8),
               xticks=np.arange(raw.shape[1]), xlabel="Layer (1 based)")
        ax.set_yticklabels(EMOTIONS)
        ax.set_xticklabels(np.arange(1, raw.shape[1] + 1), fontsize=8)
        fig.colorbar(im, ax=ax, fraction=.024, pad=.01,
                     label="AUROC difference" if title == "Denoised minus raw" else "AUROC")
    fig.suptitle(f"{label}: held-out GoEmotions one-vs-rest AUROC")
    stem = f"stage_c_goemotions_auroc_{key}"
    fig.savefig(output / f"{stem}.png", dpi=350, bbox_inches="tight", pad_inches=.15)
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight", pad_inches=.15)
    plt.close(fig)


def plot_comparison(root, output):
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    styles = (("raw", "--"), ("final", "-"))
    for (key, label), color in zip(MODELS, colors):
        folder = root / "artifacts" / "stage_c_goemotions" / key
        if not (folder / "results_metadata.json").exists():
            continue
        for kind, line_style in styles:
            matrix = np.load(folder / f"auroc_{kind}.float64.npy")
            depth = np.linspace(0, 1, matrix.shape[1])
            ax.plot(depth, matrix.mean(axis=0), color=color, ls=line_style,
                    lw=2 if kind == "final" else 1.6, alpha=1 if kind == "final" else .72,
                    label=f"{label} {kind}")
    ax.axhline(.5, color="#888888", lw=1, ls=":")
    ax.set(xlabel="Normalized layer depth", ylabel="Macro AUROC across 8 emotions",
           title="Held-out GoEmotions: raw vs denoised directions", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=.16)
    ax.legend(ncol=2, frameon=False, fontsize=9, loc="lower right")
    fig.savefig(output / "stage_c_goemotions_macro_auroc.png", dpi=350,
                bbox_inches="tight", pad_inches=.15)
    fig.savefig(output / "stage_c_goemotions_macro_auroc.svg",
                bbox_inches="tight", pad_inches=.15)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "visualizations").resolve()
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.titlesize": 14, "figure.titleweight": "bold",
                         "svg.fonttype": "none"})
    for key, label in MODELS:
        folder = root / "artifacts" / "stage_c_goemotions" / key
        if (folder / "results_metadata.json").exists():
            plot_model(folder, output, key, label)
    plot_comparison(root, output)
    print(f"Saved Stage C heatmaps and macro AUROC figure to {output}")


if __name__ == "__main__":
    main()
