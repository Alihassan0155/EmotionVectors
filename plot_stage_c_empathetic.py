"""Publication figures for matched seven-emotion cross-dataset validation."""

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
EMOTIONS = ("joy", "excitement", "gratitude", "sadness", "anger", "fear", "disgust")


def save(fig, output, stem):
    fig.savefig(output / f"{stem}.png", dpi=350, bbox_inches="tight", pad_inches=.15)
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight", pad_inches=.15)
    plt.close(fig)


def heatmaps(folder, output, key, label, selected):
    go = np.load(folder / "auroc_goemotions_matched7.float64.npy")
    ed = np.load(folder / "auroc_empathetic_final.float64.npy")
    if go.shape != ed.shape or go.shape[0] != 7:
        raise ValueError(f"Mismatched cross-dataset AUROC arrays: {key}")
    gap = go - ed
    scale = max(.1, float(np.ceil(np.max(np.abs(gap)) * 10) / 10))
    fig, axes = plt.subplots(3, 1, figsize=(max(10, go.shape[1] * .37), 8.8),
                             constrained_layout=True)
    for ax, values, title in zip(axes, (go, ed, gap),
                                 ("GoEmotions validation", "EmpatheticDialogues validation",
                                  "GoEmotions minus EmpatheticDialogues")):
        if title.startswith("GoEmotions minus"):
            image = ax.imshow(values, cmap="RdBu_r", aspect="auto", interpolation="nearest",
                              norm=TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale))
        else:
            image = ax.imshow(values, cmap="viridis", aspect="auto",
                              interpolation="nearest", vmin=0, vmax=1)
        ax.axvline(selected - 1, color="white" if not title.startswith("GoEmotions minus") else "black",
                   ls="--", lw=1.5, alpha=.85)
        ax.set(title=title, ylabel="Emotion", yticks=np.arange(7),
               xticks=np.arange(go.shape[1]), xlabel="Layer (1 based)")
        ax.set_yticklabels(EMOTIONS)
        ax.set_xticklabels(np.arange(1, go.shape[1] + 1), fontsize=8)
        fig.colorbar(image, ax=ax, fraction=.024, pad=.01,
                     label="AUROC difference" if title.startswith("GoEmotions minus") else "AUROC")
    fig.suptitle(f"{label}: external emotion validation · selected layer {selected}")
    save(fig, output, f"stage_c6_cross_dataset_auroc_{key}")


def comparison(root, output, selected_layers):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (key, label) in zip(axes.flat, MODELS):
        folder = root / "artifacts" / "stage_c_empathetic_dialogues" / key
        if not (folder / "results_metadata.json").exists():
            ax.set_visible(False)
            continue
        go = np.load(folder / "auroc_goemotions_matched7.float64.npy").mean(axis=0)
        ed = np.load(folder / "auroc_empathetic_final.float64.npy").mean(axis=0)
        depth = np.linspace(0, 1, len(go))
        ax.plot(depth, go, label="GoEmotions", color="#0072B2", lw=2)
        ax.plot(depth, ed, label="EmpatheticDialogues", color="#D55E00", lw=2)
        layer = selected_layers[key]["selected"]["layer_1_based"]
        ax.axvline(depth[layer - 1], color="#555555", ls="--", lw=1.4,
                   label=f"Selected layer {layer}")
        ax.set(title=label, xlabel="Normalized layer depth", ylabel="Macro AUROC",
               xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.16)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Seven matched emotions across datasets")
    save(fig, output, "stage_c6_cross_dataset_macro_auroc")


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
    stage = root / "artifacts" / "stage_c_empathetic_dialogues"
    selection = json.loads((stage / "selected_intervention_layers.json").read_text(encoding="utf-8"))
    for key, label in MODELS:
        if key in selection["models"]:
            heatmaps(stage / key, output, key, label,
                     selection["models"][key]["selected"]["layer_1_based"])
    comparison(root, output, selection["models"])
    print(f"Saved external validation figures to {output}")


if __name__ == "__main__":
    main()
