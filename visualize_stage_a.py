"""Create publication-ready Stage A data and result figures from saved scores."""

import argparse
import json
import os
import sqlite3
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_from_disk


DATASETS = (
    ("sst2", "train", "SST-2 train sample (n=12,000)"),
    ("sst2", "validation", "SST-2 validation (n=872)"),
    ("tweet_eval_sentiment", "test", "TweetEval sentiment test (n=12,284)"),
)
MODELS = ("0.5b", "1.5b")
TEMPLATES = ("template_1", "template_2", "template_3", "template_4")
LABELS = {
    "sst2": ("negative", "positive"),
    "tweet_eval_sentiment": ("negative", "neutral", "positive"),
}
COLORS = {"negative": "#d55e00", "neutral": "#7f7f7f", "positive": "#0072b2"}


def save_figure(fig, output_dir, name):
    fig.savefig(output_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}.png and {name}.pdf", flush=True)


def load_scores(project_root):
    path = project_root / "artifacts" / "stage_a_margins_cuda.sqlite3"
    if not path.is_file():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        frame = pd.read_sql_query("""
            SELECT model, dataset, split, template_id, row_index,
                   gold_label, predicted_label, logp_positive, logp_negative,
                   logp_neutral, margin
            FROM scores
        """, conn)
    finally:
        conn.close()
    expected = sum(n for _, _, n in (("sst2", "train", 12000),
                                     ("sst2", "validation", 872),
                                     ("tweet_eval_sentiment", "test", 12284))) * len(MODELS) * len(TEMPLATES)
    if len(frame) != expected:
        raise ValueError(f"Scoring is incomplete: {len(frame)}/{expected} rows")
    if not np.isfinite(frame[["logp_positive", "logp_negative", "margin"]].to_numpy()).all():
        raise ValueError("Non-finite score found")
    if not np.allclose(frame["margin"], frame["logp_positive"] - frame["logp_negative"], atol=1e-5):
        raise ValueError("Stored margin does not match its log-probabilities")
    return frame


def summarize_metrics(frame):
    records = []
    for (model, dataset, split, template), part in frame.groupby(
        ["model", "dataset", "split", "template_id"], sort=True
    ):
        gold = part["gold_label"].to_numpy()
        pred = part["predicted_label"].to_numpy()
        precisions, recalls, f1s = [], [], []
        for label in LABELS[dataset]:
            tp = np.sum((gold == label) & (pred == label))
            fp = np.sum((gold != label) & (pred == label))
            fn = np.sum((gold == label) & (pred != label))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        records.append({
            "model": model, "dataset": dataset, "split": split,
            "template_id": template, "examples": len(part),
            "accuracy": np.mean(gold == pred),
            "macro_f1": np.mean(f1s),
            "macro_recall": np.mean(recalls),
            "invalid_predictions": np.sum(pred == "INVALID"),
        })
    return pd.DataFrame(records)


def plot_data_overview(project_root, frame, output_dir):
    stored = {
        name: load_from_disk(str(project_root / "artifacts" / "datasets" / name))
        for name in ("sst2", "tweet_eval_sentiment")
    }
    sample = json.loads((project_root / "artifacts" / "sst2_train_12000_sample.json").read_text(encoding="utf-8"))
    indices = sample["row_indices"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    for row_number, (dataset, split, title) in enumerate(DATASETS):
        subset = frame[(frame.dataset == dataset) & (frame.split == split)
                       & (frame.model == "0.5b") & (frame.template_id == "template_1")]
        counts = subset.gold_label.value_counts().reindex(LABELS[dataset], fill_value=0)
        axes[row_number, 0].bar(counts.index, counts.values,
                                color=[COLORS[label] for label in counts.index])
        for x, value in enumerate(counts.values):
            axes[row_number, 0].text(x, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
        axes[row_number, 0].set_ylabel("Examples")
        axes[row_number, 0].set_title(f"{title} · class counts")

        column = "sentence" if dataset == "sst2" else "text"
        texts = stored[dataset][split][column]
        if dataset == "sst2" and split == "train":
            texts = [texts[index] for index in indices]
        lengths = [len(text.split()) for text in texts]
        axes[row_number, 1].hist(lengths, bins=40, color="#56b4e9", edgecolor="white")
        axes[row_number, 1].set_xlabel("Words per example")
        axes[row_number, 1].set_ylabel("Examples")
        axes[row_number, 1].set_title(f"{title} · text length")
    save_figure(fig, output_dir, "dataset_overview")


def plot_performance(metrics, output_dir):
    for metric, title in (
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("macro_recall", "Macro recall (TweetEval official metric)"),
    ):
        fig, axes = plt.subplots(1, len(DATASETS), figsize=(5.3 * len(DATASETS), 4.3),
                                 constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, (dataset, split, name) in zip(axes, DATASETS):
            subset = metrics[(metrics.dataset == dataset) & (metrics.split == split)]
            matrix = subset.pivot(index="model", columns="template_id", values=metric)
            values = matrix.reindex(index=MODELS, columns=TEMPLATES).to_numpy()
            im = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax.set_xticks(range(4), ["T1", "T2", "T3", "T4"])
            ax.set_yticks(range(2), ["0.5B", "1.5B"])
            ax.set_title(name)
            for y in range(2):
                for x in range(4):
                    ax.text(x, y, f"{values[y, x]:.1%}", ha="center", va="center",
                            color="white" if values[y, x] < 0.6 else "black", fontsize=10)
        fig.colorbar(im, ax=axes, shrink=0.75, label=title)
        fig.suptitle(f"Qwen2.5 sentiment classification · {title}", fontsize=14)
        save_figure(fig, output_dir, f"heatmap_{metric}")


def plot_confusions(frame, output_dir):
    for dataset, split, name in DATASETS:
        labels = LABELS[dataset]
        pred_labels = (*labels, "INVALID")
        fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), constrained_layout=True)
        for y, model in enumerate(MODELS):
            for x, template in enumerate(TEMPLATES):
                part = frame[(frame.dataset == dataset) & (frame.split == split)
                             & (frame.model == model) & (frame.template_id == template)]
                table = pd.crosstab(part.gold_label, part.predicted_label)
                counts = table.reindex(index=labels, columns=pred_labels, fill_value=0).to_numpy()
                normalized = counts / counts.sum(axis=1, keepdims=True)
                ax = axes[y, x]
                im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues", aspect="auto")
                ax.set_title(f"{model} · {template.replace('template_', 'T')}")
                ax.set_xticks(range(len(pred_labels)), pred_labels, rotation=45, ha="right")
                ax.set_yticks(range(len(labels)), labels)
                if x == 0:
                    ax.set_ylabel("Gold label")
                if y == 1:
                    ax.set_xlabel("Predicted label")
                for row in range(len(labels)):
                    for col in range(len(pred_labels)):
                        value = normalized[row, col]
                        ax.text(col, row, f"{value:.0%}\n{counts[row, col]:,}",
                                ha="center", va="center", fontsize=7,
                                color="white" if value > 0.5 else "black")
        fig.colorbar(im, ax=axes, shrink=0.68, label="Fraction of gold class")
        fig.suptitle(f"Confusion matrices · {name}", fontsize=14)
        save_figure(fig, output_dir, f"confusion_{dataset}_{split}")


def plot_margins(frame, output_dir):
    for dataset, split, name in DATASETS:
        subset = frame[(frame.dataset == dataset) & (frame.split == split)]
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True,
                                 sharey=True, constrained_layout=True)
        for y, model in enumerate(MODELS):
            for x, template in enumerate(TEMPLATES):
                ax = axes[y, x]
                part = subset[(subset.model == model) & (subset.template_id == template)]
                for label in LABELS[dataset]:
                    values = part.loc[part.gold_label == label, "margin"].to_numpy()
                    ax.hist(values, bins=65, range=(-18, 18), density=True,
                            histtype="step", linewidth=1.8, label=label,
                            color=COLORS[label])
                ax.axvline(0, color="black", linestyle="--", linewidth=0.9)
                ax.set_title(f"{model} · {template.replace('template_', 'T')}")
                if x == 0:
                    ax.set_ylabel("Density")
                if y == 1:
                    ax.set_xlabel("M = log P(positive) − log P(negative)")
                ax.set_xlim(-18, 18)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(f"Gold-label margin distributions · {name}", y=1.03, fontsize=14)
        save_figure(fig, output_dir, f"margins_{dataset}_{split}")


def plot_mean_margin_heatmap(frame, output_dir):
    means = frame.groupby(
        ["model", "dataset", "split", "template_id", "gold_label"]
    )["margin"].mean()
    scale = max(abs(means.min()), abs(means.max()))
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(5.3 * len(DATASETS), 6.5),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (dataset, split, name) in zip(axes, DATASETS):
        row_names = [(model, label) for model in MODELS for label in LABELS[dataset]]
        values = np.array([
            [means[(model, dataset, split, template, label)] for template in TEMPLATES]
            for model, label in row_names
        ])
        im = ax.imshow(values, vmin=-scale, vmax=scale, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(4), ["T1", "T2", "T3", "T4"])
        ax.set_yticks(range(len(row_names)), [f"{model} · {label}" for model, label in row_names])
        ax.set_title(name)
        for row in range(len(row_names)):
            for col in range(4):
                value = values[row, col]
                ax.text(col, row, f"{value:+.2f}", ha="center", va="center",
                        color="white" if abs(value) > scale * 0.52 else "black", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.76,
                 label="Mean M = log P(positive) − log P(negative)")
    fig.suptitle("Mean first-token sentiment margin by gold class", fontsize=14)
    save_figure(fig, output_dir, "heatmap_mean_margin")


def plot_prompt_disagreement(frame, output_dir):
    matrices = {}
    for model in MODELS:
        for dataset, split, _ in DATASETS:
            part = frame[(frame.model == model) & (frame.dataset == dataset)
                         & (frame.split == split)]
            wide = part.pivot(index="row_index", columns="template_id", values="predicted_label")
            values = wide.reindex(columns=TEMPLATES).to_numpy()
            matrices[(model, dataset, split)] = np.array([
                [np.mean(values[:, i] != values[:, j]) for j in range(4)]
                for i in range(4)
            ])
    maximum = max(matrix.max() for matrix in matrices.values())
    fig, axes = plt.subplots(2, len(DATASETS),
                             figsize=(4.5 * len(DATASETS), 8), constrained_layout=True)
    for y, model in enumerate(MODELS):
        for x, (dataset, split, name) in enumerate(DATASETS):
            values = matrices[(model, dataset, split)]
            ax = axes[y, x]
            im = ax.imshow(values, vmin=0, vmax=maximum, cmap="magma")
            ax.set_xticks(range(4), ["T1", "T2", "T3", "T4"])
            ax.set_yticks(range(4), ["T1", "T2", "T3", "T4"])
            ax.set_title(f"{model} · {name}")
            for row in range(4):
                for col in range(4):
                    ax.text(col, row, f"{values[row, col]:.0%}", ha="center", va="center",
                            color="white" if values[row, col] < maximum / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.75, label="Fraction of examples with different predictions")
    fig.suptitle("Prompt sensitivity · pairwise label disagreement", fontsize=14)
    save_figure(fig, output_dir, "heatmap_prompt_disagreement")


def write_report(frame, metrics, output_dir):
    margin_summary = frame.groupby(
        ["model", "dataset", "split", "template_id", "gold_label"]
    )["margin"].agg(["count", "mean", "median", "std"]).reset_index()
    margin_summary.to_csv(output_dir / "margin_summary.csv", index=False)
    metrics.to_csv(output_dir / "performance_metrics.csv", index=False)
    lines = [
        "# Stage A figures",
        "",
        "Source: the completed `stage_a_margins_cuda.sqlite3` checkpoint. "
        "All 201,248 scored rows are included. SST-2 train sample and validation "
        "are separate; TweetEval uses its sentiment test split.",
        "",
        "- `dataset_overview`: class counts and word-length distributions.",
        "- `heatmap_accuracy`, `heatmap_macro_f1`, `heatmap_macro_recall`: "
        "model-by-template performance; TweetEval's official metric is macro recall.",
        "- `confusion_*`: row-normalized class errors for every model and prompt.",
        "- `margins_*`: positive-minus-negative log-probability margins by gold class. "
        "TweetEval neutral is shown as its own gold class; M alone does not classify neutral.",
        "- `heatmap_mean_margin`: average margin for each gold class, model, and template.",
        "- `heatmap_prompt_disagreement`: fraction of examples whose predicted labels "
        "change between each pair of fixed templates.",
        "- `performance_metrics.csv` and `margin_summary.csv`: numeric values behind the figures.",
        "",
        "Each plot is exported as PNG and vector PDF. T1–T4 refer to the fixed "
        "templates in `prompt_templates.py`. `INVALID` predictions count as errors.",
        "",
        "## Accuracy range across templates",
        "",
    ]
    for dataset, split, name in DATASETS:
        for model in MODELS:
            subset = metrics[(metrics.dataset == dataset) & (metrics.split == split)
                             & (metrics.model == model)]
            low, high = subset.accuracy.min(), subset.accuracy.max()
            lines.append(f"- {name}, {model}: {low:.1%}–{high:.1%} "
                         f"({(high-low)*100:.1f} percentage-point spread).")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = (args.output or project_root / "artifacts" / "figures").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    frame = load_scores(project_root)
    metrics = summarize_metrics(frame)
    plot_data_overview(project_root, frame, output_dir)
    plot_performance(metrics, output_dir)
    plot_confusions(frame, output_dir)
    plot_margins(frame, output_dir)
    plot_mean_margin_heatmap(frame, output_dir)
    plot_prompt_disagreement(frame, output_dir)
    write_report(frame, metrics, output_dir)
    print(f"Finished {len(frame):,} scored rows; figures and tables: {output_dir}")


if __name__ == "__main__":
    main()
