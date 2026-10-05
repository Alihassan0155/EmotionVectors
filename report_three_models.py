"""Validate and combine completed Qwen and BERT Stage A results and plots."""

import argparse
import json
import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bert_stage_a import TEMPLATES as BERT_TEMPLATES
from prompt_templates import PROMPT_TEMPLATES


MODELS = ("qwen2.5-0.5b", "qwen2.5-1.5b", "bert-large-uncased")
MODEL_NAMES = ("Qwen 0.5B", "Qwen 1.5B", "BERT-Large")
DATASETS = (("sst2", "combined", 12872, ("negative", "positive")),
            ("tweet_eval_sentiment", "test", 12284, ("negative", "neutral", "positive")))
TEMPLATES = tuple(BERT_TEMPLATES)
COLORS = {"negative": "#d55e00", "neutral": "#777777", "positive": "#0072b2"}


def read_table(path, query):
    if not path.is_file():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


def load_results(source_root):
    qwen_path = source_root / "artifacts" / "pooled_stage_a" / "qwen_stage_a_pooled_scores.sqlite3"
    bert_path = source_root / "artifacts" / "bert_stage_a" / "scores.sqlite3"
    qwen = read_table(qwen_path, "SELECT * FROM scores")
    qwen["model"] = qwen["model"].map({"0.5b": MODELS[0], "1.5b": MODELS[1]})
    qwen["prompt_family"] = "Qwen instruction"
    qwen["scoring_method"] = "causal next-token"
    bert = read_table(bert_path, "SELECT * FROM scores")
    bert["model"] = MODELS[2]
    bert["split"] = bert["dataset"].map({"sst2": "combined", "tweet_eval_sentiment": "test"})
    bert["prompt_family"] = "BERT cloze"
    bert["scoring_method"] = "masked token"
    if qwen.model.isna().any() or bert.split.isna().any():
        raise ValueError("Unknown source model or dataset")
    fields = ("model", "dataset", "split", "template_id", "row_index", "source_split",
              "source_row_index", "gold_label", "predicted_label", "logp_positive",
              "logp_negative", "logp_neutral", "margin", "prompt_family", "scoring_method")
    frame = pd.concat((qwen[list(fields)], bert[list(fields)]), ignore_index=True)
    expected = sum(item[2] for item in DATASETS) * len(MODELS) * len(TEMPLATES)
    if len(frame) != expected:
        raise ValueError(f"Incomplete score set: {len(frame):,}/{expected:,}")
    key = ("model", "dataset", "split", "template_id", "row_index")
    if frame.duplicated(list(key)).any():
        raise ValueError("Duplicate model/dataset/template/row scores")
    for model in MODELS:
        for dataset, split, count, labels in DATASETS:
            for template in TEMPLATES:
                part = frame[(frame.model == model) & (frame.dataset == dataset)
                             & (frame.split == split) & (frame.template_id == template)]
                if len(part) != count or set(part.gold_label) != set(labels):
                    raise ValueError(f"Incomplete result: {model}/{dataset}/{template}")
                if not set(part.predicted_label).issubset(set(labels) | {"INVALID"}):
                    raise ValueError(f"Unexpected prediction: {model}/{dataset}/{template}")
    numbers = frame[["logp_positive", "logp_negative", "margin"]].to_numpy(float)
    if not np.isfinite(numbers).all() or not np.allclose(numbers[:, 0] - numbers[:, 1], numbers[:, 2], atol=1e-4):
        raise ValueError("Non-finite or inconsistent sentiment margin")
    if frame[frame.dataset == "tweet_eval_sentiment"].logp_neutral.isna().any():
        raise ValueError("Missing neutral log-probability")
    # The two experiments must evaluate exactly the same examples and labels.
    identity = ["dataset", "split", "template_id", "row_index", "source_split", "source_row_index", "gold_label"]
    q_identity = qwen[identity].drop_duplicates()
    b_identity = bert[identity].drop_duplicates()
    if len(q_identity) != len(b_identity) or len(q_identity.merge(b_identity, how="inner", on=identity)) != len(b_identity):
        raise ValueError("Qwen and BERT checkpoints do not align on examples and gold labels")
    return frame


def calculate_metrics(frame):
    rows = []
    for (model, dataset, split, template), part in frame.groupby(
        ["model", "dataset", "split", "template_id"], sort=True
    ):
        labels = next(item[3] for item in DATASETS if item[0] == dataset)
        gold, pred = part.gold_label.to_numpy(), part.predicted_label.to_numpy()
        recalls, f1s = [], []
        for label in labels:
            tp = np.sum((gold == label) & (pred == label))
            actual = np.sum(gold == label)
            predicted = np.sum(pred == label)
            recalls.append(tp / actual if actual else 0.0)
            f1s.append(2 * tp / (actual + predicted) if actual + predicted else 0.0)
        invalid = np.sum(pred == "INVALID")
        rows.append((model, dataset, split, template, len(part), np.mean(gold == pred),
                     np.mean(f1s), np.mean(recalls), invalid, invalid / len(part)))
    return pd.DataFrame(rows, columns=("model", "dataset", "split", "template_id", "examples",
                                       "accuracy", "macro_f1", "macro_recall",
                                       "invalid_predictions", "invalid_rate"))


def save_figure(fig, folder, name):
    fig.savefig(folder / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def heatmap_performance(metrics, folder):
    for metric, title in (("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"),
                          ("macro_recall", "Macro recall")):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        for ax, (dataset, split, _, _) in zip(axes, DATASETS):
            part = metrics[(metrics.dataset == dataset) & (metrics.split == split)]
            matrix = part.pivot(index="model", columns="template_id", values=metric)
            values = matrix.reindex(index=MODELS, columns=TEMPLATES).to_numpy()
            im = ax.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax.set_xticks(range(4), ["T1", "T2", "T3", "T4"])
            ax.set_yticks(range(3), MODEL_NAMES)
            ax.set_title("SST-2 pooled" if dataset == "sst2" else "TweetEval test")
            for y in range(3):
                for x in range(4):
                    ax.text(x, y, f"{values[y,x]:.1%}", ha="center", va="center",
                            color="white" if values[y,x] < 0.6 else "black")
        fig.colorbar(im, ax=axes, shrink=0.75, label=title)
        fig.suptitle(f"Three-model sentiment baseline: {title}")
        save_figure(fig, folder, f"heatmap_{metric}")


def confusion_figures(frame, folder):
    for dataset, split, _, labels in DATASETS:
        fig, axes = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
        for y, model in enumerate(MODELS):
            for x, template in enumerate(TEMPLATES):
                part = frame[(frame.model == model) & (frame.dataset == dataset)
                             & (frame.template_id == template)]
                count = pd.crosstab(part.gold_label, part.predicted_label).reindex(
                    index=labels, columns=(*labels, "INVALID"), fill_value=0).to_numpy()
                values = count / count.sum(axis=1, keepdims=True)
                ax = axes[y, x]
                im = ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
                ax.set_title(f"{MODEL_NAMES[y]} / T{x+1}")
                ax.set_xticks(range(len(labels)+1), (*labels, "INVALID"), rotation=35, ha="right")
                ax.set_yticks(range(len(labels)), labels)
                for row in range(len(labels)):
                    for col in range(len(labels)+1):
                        ax.text(col, row, f"{values[row,col]:.0%}", ha="center", va="center",
                                fontsize=8, color="white" if values[row,col] > 0.5 else "black")
        fig.colorbar(im, ax=axes, shrink=0.7, label="Fraction of gold class")
        fig.suptitle(f"Confusion matrices: {dataset}")
        save_figure(fig, folder, f"confusion_{dataset}")


def margin_figures(frame, folder):
    for dataset, split, _, labels in DATASETS:
        fig, axes = plt.subplots(3, 4, figsize=(15, 10), sharex=True, constrained_layout=True)
        for y, model in enumerate(MODELS):
            for x, template in enumerate(TEMPLATES):
                part = frame[(frame.model == model) & (frame.dataset == dataset)
                             & (frame.template_id == template)]
                ax = axes[y, x]
                for label in labels:
                    values = part.loc[part.gold_label == label, "margin"].to_numpy()
                    ax.hist(values, bins=60, range=(-18, 18), density=True, histtype="step",
                            linewidth=1.5, color=COLORS[label], label=label)
                ax.axvline(0, color="black", linestyle="--", linewidth=0.7)
                ax.set_xlim(-18, 18)
                ax.set_title(f"{MODEL_NAMES[y]} / T{x+1}")
                if x == 0:
                    ax.set_ylabel("Density")
                if y == 2:
                    ax.set_xlabel("M = log P(pos) - log P(neg)")
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, names, loc="lower center", ncol=len(names), bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(f"Gold-class margin distributions: {dataset}")
        save_figure(fig, folder, f"margins_{dataset}")


def disagreement_figure(frame, folder):
    fig, axes = plt.subplots(3, 2, figsize=(9, 12), constrained_layout=True)
    for y, model in enumerate(MODELS):
        for x, (dataset, split, _, _) in enumerate(DATASETS):
            part = frame[(frame.model == model) & (frame.dataset == dataset)]
            wide = part.pivot(index="row_index", columns="template_id", values="predicted_label")
            values = wide.reindex(columns=TEMPLATES).to_numpy()
            matrix = np.array([[np.mean(values[:, i] != values[:, j]) for j in range(4)]
                               for i in range(4)])
            ax = axes[y, x]
            im = ax.imshow(matrix, vmin=0, vmax=1, cmap="magma")
            ax.set_xticks(range(4), ["T1", "T2", "T3", "T4"])
            ax.set_yticks(range(4), ["T1", "T2", "T3", "T4"])
            ax.set_title(f"{MODEL_NAMES[y]} / {dataset}")
            for row in range(4):
                for col in range(4):
                    ax.text(col, row, f"{matrix[row,col]:.0%}", ha="center", va="center",
                            color="white" if matrix[row,col] < 0.5 else "black")
    fig.colorbar(im, ax=axes, shrink=0.7, label="Prediction disagreement")
    fig.suptitle("Sensitivity to four fixed prompts")
    save_figure(fig, folder, "heatmap_prompt_disagreement")


def mean_margin_figure(frame, folder):
    means = frame.groupby(["model", "dataset", "template_id", "gold_label"]).margin.mean()
    scale = max(abs(means.min()), abs(means.max()))
    fig, axes = plt.subplots(1, 2, figsize=(11, 7), constrained_layout=True)
    for ax, (dataset, _, _, labels) in zip(axes, DATASETS):
        pairs = [(model, label) for model in MODELS for label in labels]
        matrix = np.array([[means[(model, dataset, template, label)] for template in TEMPLATES]
                           for model, label in pairs])
        im = ax.imshow(matrix, vmin=-scale, vmax=scale, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(4), ["T1", "T2", "T3", "T4"])
        ax.set_yticks(range(len(pairs)), [f"{model} / {label}" for model, label in pairs])
        ax.set_title(dataset)
        for row in range(len(pairs)):
            for col in range(4):
                ax.text(col, row, f"{matrix[row,col]:+.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(matrix[row,col]) > scale * 0.5 else "black")
    fig.colorbar(im, ax=axes, shrink=0.7, label="Mean positive-minus-negative log-probability")
    fig.suptitle("Mean sentiment margin by gold class")
    save_figure(fig, folder, "heatmap_mean_margin")


def save_results(frame, metrics, folder):
    folder.mkdir(parents=True, exist_ok=True)
    db_path = folder / "scores.sqlite3"
    temporary = folder / "scores.sqlite3.tmp"
    if temporary.exists():
        temporary.unlink()
    conn = sqlite3.connect(temporary)
    try:
        frame.to_sql("scores", conn, index=False, chunksize=5000)
        conn.execute("CREATE UNIQUE INDEX score_key ON scores(model,dataset,split,template_id,row_index)")
        conn.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        for dataset, prompts in PROMPT_TEMPLATES.items():
            for template, prompt in prompts.items():
                conn.execute("INSERT INTO metadata VALUES (?,?)", (f"qwen/{dataset}/{template}", prompt))
        for template, prompt in BERT_TEMPLATES.items():
            conn.execute("INSERT INTO metadata VALUES (?,?)", (f"bert/{template}", prompt))
        conn.commit()
    finally:
        conn.close()
    temporary.replace(db_path)
    metrics.to_csv(folder / "performance_metrics.csv", index=False)
    margin = frame.groupby(["model", "dataset", "split", "template_id", "gold_label"]).margin.agg(
        count="count", mean="mean", std="std", min="min", p05=lambda x: x.quantile(0.05),
        p25=lambda x: x.quantile(0.25), median="median", p75=lambda x: x.quantile(0.75),
        p95=lambda x: x.quantile(0.95), max="max").reset_index()
    margin.to_csv(folder / "margin_summary.csv", index=False)
    lines = ["STAGE A: THREE-MODEL ZERO-SHOT SENTIMENT BASELINE", "",
             "Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, google-bert/bert-large-uncased.",
             "SST-2 pooled: 12,000 sampled train + 872 validation = 12,872 rows.",
             "TweetEval sentiment: 12,284 test rows. Four fixed prompts per model.",
             "Source split and row are preserved; SST-2 pooled is not official validation.",
             "Qwen uses instruction prompts and causal next-token label scores; BERT uses",
             "cloze prompts and masked-token label scores. T1-T4 are different wordings",
             "across model families, so corresponding columns are not identical prompts.",
             "Accuracy/F1 are descriptive comparisons, not controlled architecture effects.",
             "M = log P(positive) - log P(negative), computed within each model's scoring",
             "objective. Raw M scales should not be compared directly across model families.",
             "TweetEval's official sentiment metric is macro-recall.", "",
             "Each entry below is accuracy / macro-F1 / macro-recall, in percent.",
             "Invalid-output rate = INVALID generated labels / all examples.",
             "BERT selects among allowed label tokens, so its invalid rate is 0 by design.", ""]
    for dataset, split, _, _ in DATASETS:
        lines.append(f"{dataset} ({split})")
        for model in MODELS:
            cells = []
            for template in TEMPLATES:
                row = metrics[(metrics.dataset == dataset) & (metrics.model == model)
                              & (metrics.template_id == template)].iloc[0]
                cells.append(f"{template.replace('template_', 'T')} "
                             f"{row.accuracy*100:.2f}/{row.macro_f1*100:.2f}/{row.macro_recall*100:.2f}")
            lines.append(f"  {model}: " + "  ".join(cells))
        lines.append("")
    lines.append("Invalid-output rate by dataset / model / template (percent; count in parentheses)")
    for dataset, split, _, _ in DATASETS:
        lines.append(f"{dataset} ({split})")
        for model in MODELS:
            cells = []
            for template in TEMPLATES:
                row = metrics[(metrics.dataset == dataset) & (metrics.model == model)
                              & (metrics.template_id == template)].iloc[0]
                cells.append(f"{template.replace('template_', 'T')} "
                             f"{row.invalid_rate*100:.3f}% ({int(row.invalid_predictions)})")
            lines.append(f"  {model}: " + "  ".join(cells))
        lines.append("")
    lines += ["All per-example scores: scores.sqlite3", "Tables: performance_metrics.csv, margin_summary.csv",
              "Margin table includes per-class 5th/25th/50th/75th/95th percentiles.",
              "Full M distributions by model/prompt/gold class: figures/margins_*.png and .pdf",
              "Other figures: figures/*.png and figures/*.pdf"]
    (folder / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output = (args.output or source_root / "artifacts" / "stage_a_three_models").resolve()
    frame = load_results(source_root)
    metrics = calculate_metrics(frame)
    save_results(frame, metrics, output)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    heatmap_performance(metrics, figures)
    confusion_figures(frame, figures)
    margin_figures(frame, figures)
    disagreement_figure(frame, figures)
    mean_margin_figure(frame, figures)
    print(f"Verified {len(frame):,} scores across {len(MODELS)} models; output: {output}")


if __name__ == "__main__":
    main()
