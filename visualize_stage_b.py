"""Publication figures for completed Stage B raw emotion directions.

Reads only the small exported direction arrays and metadata. The pooled
activation checkpoints and experiment outputs are never opened for writing.
"""

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
from sklearn.decomposition import PCA
from umap import UMAP


MODELS = (
    ("0.5b", "Qwen 0.5B"),
    ("1.5b", "Qwen 1.5B"),
    ("3b", "Qwen 3B"),
    ("bert_large_uncased", "BERT-large"),
)
COLORS = ("#1f77b4", "#d95f02", "#2ca02c", "#9467bd",
          "#e377c2", "#8c564b", "#17becf", "#bcbd22")


def load_vectors(root: Path):
    output = {}
    source_hashes = set()
    b2_hashes = set()
    for key, label in MODELS:
        folder = root / "artifacts" / "stage_b_raw_vectors" / key
        meta = json.loads((folder / "vectors_metadata.json").read_text(encoding="utf-8"))
        vectors = np.load(folder / "emotion_directions_raw.float32.npy")
        expected = (8, meta["layers"], meta["hidden_size"])
        if not meta.get("complete") or vectors.shape != expected or not np.isfinite(vectors).all():
            raise ValueError(f"Incomplete or invalid vectors: {folder}")
        if meta.get("neutral_pca_status", "").find("pending") < 0:
            raise ValueError(f"Expected raw, pre-denoising directions: {folder}")
        source_hashes.add(meta["source_sha256"])
        b2_hashes.add(meta["b2_vector_train_sha256"])
        output[key] = {"label": label, "meta": meta, "vectors": vectors}
    if len(source_hashes) != 1 or len(b2_hashes) != 1:
        raise ValueError("Model vectors were not constructed from the same B2 training set")
    if any(item["meta"]["emotion_order"] != output["0.5b"]["meta"]["emotion_order"]
           for item in output.values()):
        raise ValueError("Emotion ordering differs between models")
    return output


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10,
        "figure.titlesize": 15, "figure.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "savefig.facecolor": "white",
        "svg.fonttype": "none",
    })


def save(fig, output: Path, stem: str):
    fig.savefig(output / f"{stem}.png", dpi=350, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def unit_vectors(vectors):
    norms = np.linalg.norm(vectors.astype(np.float64), axis=-1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Zero-length emotion direction")
    return vectors / norms


def norm_figure(data, output):
    emotions = data["0.5b"]["meta"]["emotion_order"]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.3), constrained_layout=True)
    for ax, (key, _) in zip(axes.flat, MODELS):
        item = data[key]
        values = np.linalg.norm(item["vectors"].astype(np.float64), axis=2)
        x = np.arange(1, values.shape[1] + 1)
        for i, emotion in enumerate(emotions):
            ax.plot(x, values[i], color=COLORS[i], label=emotion, lw=1.8, alpha=.95)
        ax.set(title=item["label"], xlabel="Layer (1 based)", ylabel="Raw direction L2 norm",
               xlim=(1, len(x)))
        ax.grid(alpha=.17)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=8, frameon=False)
    fig.suptitle("Emotion direction magnitude across layers")
    save(fig, output, "stage_b_norm_by_layer")


def cosine_figure(data, output):
    emotions = data["0.5b"]["meta"]["emotion_order"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.8), constrained_layout=True)
    representative = {}
    for ax, (key, _) in zip(axes.flat, MODELS):
        item = data[key]
        layers = item["meta"]["layers"]
        layer = round((2 / 3) * (layers - 1))  # zero based; fixed normalized-depth rule
        representative[key] = {"layer_1_based": layer + 1,
                               "normalized_depth": layer / (layers - 1)}
        u = unit_vectors(item["vectors"][:, layer, :])
        matrix = np.clip(u @ u.T, -1, 1)
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set(title=f"{item['label']}  ·  layer {layer + 1}/{layers}",
               xticks=np.arange(8), yticks=np.arange(8))
        ax.set_xticklabels(emotions, rotation=45, ha="right")
        ax.set_yticklabels(emotions)
        for i in range(8):
            for j in range(8):
                value = matrix[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(value) > .57 else "#17212b")
        ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=axes, shrink=.7, label="Cosine similarity")
    fig.suptitle("Pairwise cosine similarity at two-thirds model depth")
    save(fig, output, "stage_b_cosine_heatmaps")
    return representative


def embedding_panel(ax, coords, emotions, title, variance=None):
    for i, emotion in enumerate(emotions):
        ax.scatter(coords[i, 0], coords[i, 1], s=105, color=COLORS[i],
                   edgecolor="white", linewidth=.8, zorder=3, label=emotion)
    ax.set_title(title)
    if variance is None:
        ax.set(xlabel="UMAP 1", ylabel="UMAP 2")
    else:
        ax.set(xlabel=f"PC1 ({variance[0]:.1%} variance)",
               ylabel=f"PC2 ({variance[1]:.1%} variance)")
    ax.margins(.27)
    ax.grid(alpha=.15)


def embedding_figure(data, output, representative):
    emotions = data["0.5b"]["meta"]["emotion_order"]
    fig, axes = plt.subplots(4, 2, figsize=(12.5, 18), constrained_layout=True)
    embedding_info = {}
    for row, (key, _) in enumerate(MODELS):
        item = data[key]
        layer = representative[key]["layer_1_based"] - 1
        vectors = unit_vectors(item["vectors"][:, layer, :])
        pca = PCA(n_components=2, svd_solver="full")
        pc = pca.fit_transform(vectors)
        um = UMAP(n_components=2, n_neighbors=3, min_dist=.1, metric="cosine",
                  random_state=42, n_jobs=1, init="random")
        embedding = um.fit_transform(vectors)
        embedding_panel(axes[row, 0], pc, emotions,
                        f"{item['label']} · PCA · layer {layer + 1}",
                        pca.explained_variance_ratio_)
        embedding_panel(axes[row, 1], embedding, emotions,
                        f"{item['label']} · UMAP · layer {layer + 1}")
        embedding_info[key] = {"pca_explained_variance_ratio":
                               pca.explained_variance_ratio_.tolist()}
    fig.suptitle("Eight emotion directions in two dimensions\n"
                 "Unit-normalized raw vectors; each model embedded separately")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=8, frameon=False)
    save(fig, output, "stage_b_pca_umap")
    return embedding_info


def comparison_figure(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), constrained_layout=True)
    model_colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    for (key, _), color in zip(MODELS, model_colors):
        item = data[key]
        vectors = item["vectors"].astype(np.float64)
        depth = np.linspace(0, 1, vectors.shape[1])
        norms = np.linalg.norm(vectors, axis=2)
        median = np.median(norms)
        axes[0].plot(depth, norms.mean(axis=0) / median, label=item["label"],
                     color=color, lw=2.2)
        u = unit_vectors(vectors)
        cosines = np.einsum("eld,fld->lef", u, u)
        mask = ~np.eye(8, dtype=bool)
        mean_abs_cos = np.abs(cosines[:, mask]).mean(axis=1)
        axes[1].plot(depth, mean_abs_cos, label=item["label"], color=color, lw=2.2)
    axes[0].set(xlabel="Normalized layer depth", ylabel="Mean norm / model median norm",
                title="Relative magnitude across depth", xlim=(0, 1))
    axes[1].set(xlabel="Normalized layer depth", ylabel="Mean absolute off-diagonal cosine",
                title="Emotion direction alignment", xlim=(0, 1), ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=.17)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Cross-model layer trends (within-model quantities)")
    save(fig, output, "stage_b_cross_model_normalized_depth")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "visualizations").resolve()
    data = load_vectors(root)
    output.mkdir(parents=True, exist_ok=True)
    style()
    norm_figure(data, output)
    representative = cosine_figure(data, output)
    embedding_info = embedding_figure(data, output, representative)
    comparison_figure(data, output)
    manifest = {
        "source": "Stage B raw directions before neutral PCA denoising",
        "source_formula": "mu_emotion - mean(mu_other_7_emotions)",
        "examples_per_model": 7417,
        "representative_layer_rule": "round(2/3 * (layers - 1)) + 1; fixed before inspection",
        "representative_layers": representative,
        "embedding_input": "eight unit-normalized directions at representative layer, per model",
        "umap": {"n_neighbors": 3, "min_dist": .1, "metric": "cosine",
                 "random_state": 42, "interpretation": "exploratory: only eight points per model"},
        "cross_model_norm": "mean emotion norm / median norm over all emotions and layers within model",
        "cross_model_similarity": "mean absolute off-diagonal cosine across 28 emotion pairs",
        "pca": embedding_info,
        "models": {key: {"model_id": item["meta"]["model_id"],
                         "layers": item["meta"]["layers"],
                         "hidden_size": item["meta"]["hidden_size"]}
                   for key, item in data.items()},
        "files": [f"{stem}.{extension}" for stem in (
            "stage_b_norm_by_layer", "stage_b_cosine_heatmaps",
            "stage_b_pca_umap", "stage_b_cross_model_normalized_depth")
                  for extension in ("png", "svg")],
    }
    (output / "stage_b_plot_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(manifest['files'])} figures and manifest to {output}")


if __name__ == "__main__":
    main()
