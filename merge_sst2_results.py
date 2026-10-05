"""Pool the fixed SST-2 sample and validation set, then rebuild Qwen results."""

import argparse
import hashlib
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
from datasets import ClassLabel, Dataset, Features, Value, load_from_disk

import visualize_stage_a as charts

SST_COUNT = 12_872
TWEET_COUNT = 12_284


def make_combined_dataset(source_root, output_dir):
    source = load_from_disk(str(source_root / "artifacts" / "datasets" / "sst2"))
    manifest_path = source_root / "artifacts" / "sst2_train_12000_sample.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_indices = manifest["row_indices"]
    if len(train_indices) != 12_000 or len(set(train_indices)) != 12_000:
        raise ValueError("The saved SST-2 training sample is not 12,000 unique rows")
    if (manifest["train_fingerprint"] != source["train"]._fingerprint or
            manifest["validation_fingerprint"] != source["validation"]._fingerprint):
        raise ValueError("SST-2 dataset fingerprints differ from the sample manifest")

    train = source["train"]
    validation = source["validation"]
    texts = [train[i]["sentence"] for i in train_indices]
    texts.extend(validation["sentence"])
    labels = [train[i]["label"] for i in train_indices]
    labels.extend(validation["label"])
    if len(texts) != SST_COUNT or set(texts[:12_000]) & set(texts[12_000:]):
        raise ValueError("SST-2 pooling failed or the two sources have text overlap")
    features = Features({
        "text": Value("string"),
        "label": ClassLabel(names=["negative", "positive"]),
        "source_split": Value("string"),
        "source_row_index": Value("int64"),
        "combined_index": Value("int64"),
    })
    combined = Dataset.from_dict({
        "text": texts,
        "label": labels,
        "source_split": ["train"] * 12_000 + ["validation"] * 872,
        "source_row_index": train_indices + list(range(872)),
        "combined_index": list(range(SST_COUNT)),
    }, features=features)
    dataset_dir = output_dir / "sst2_combined_12872"
    if dataset_dir.exists():
        existing = load_from_disk(str(dataset_dir))
        if (len(existing) != len(combined) or
                existing["source_row_index"] != combined["source_row_index"] or
                existing["source_split"] != combined["source_split"] or
                existing["label"] != combined["label"]):
            raise ValueError(f"Existing combined dataset differs: {dataset_dir}")
    else:
        combined.save_to_disk(str(dataset_dir))
    return combined, train_indices, hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def merge_scores(source_root, train_indices):
    score_path = source_root / "artifacts" / "stage_a_margins_cuda.sqlite3"
    conn = sqlite3.connect(f"file:{score_path.as_posix()}?mode=ro", uri=True)
    try:
        frame = pd.read_sql_query("""
            SELECT model, dataset, split, template_id, row_index,
                   gold_label, predicted_label, logp_positive, logp_negative,
                   logp_neutral, margin
            FROM scores
        """, conn)
    finally:
        conn.close()
    if len(frame) != 201_248:
        raise ValueError(f"Expected 201,248 scored rows, found {len(frame)}")

    index_map = pd.DataFrame({
        "split": ["train"] * 12_000 + ["validation"] * 872,
        "row_index": train_indices + list(range(872)),
        "combined_index": list(range(SST_COUNT)),
    })
    sst = frame[frame.dataset == "sst2"].merge(
        index_map, on=["split", "row_index"], how="left", validate="many_to_one"
    )
    if sst.combined_index.isna().any() or len(sst) != SST_COUNT * 8:
        raise ValueError("SST-2 scores do not match the combined row map")
    sst["source_split"] = sst["split"]
    sst["source_row_index"] = sst["row_index"]
    sst["row_index"] = sst["combined_index"].astype("int64")
    sst["split"] = "combined"
    sst = sst.drop(columns="combined_index")

    tweet = frame[frame.dataset == "tweet_eval_sentiment"].copy()
    if len(tweet) != TWEET_COUNT * 8 or set(tweet["split"]) != {"test"}:
        raise ValueError("TweetEval source rows are incomplete")
    tweet["source_split"] = tweet["split"]
    tweet["source_row_index"] = tweet["row_index"]
    merged = pd.concat([sst, tweet], ignore_index=True)
    expected = {("sst2", "combined"): SST_COUNT,
                ("tweet_eval_sentiment", "test"): TWEET_COUNT}
    for (model, dataset, split, template), rows in merged.groupby(
        ["model", "dataset", "split", "template_id"]
    ):
        if len(rows) != expected[(dataset, split)] or rows.row_index.nunique() != len(rows):
            raise ValueError(f"Incomplete or repeated merged result: {(model, dataset, split, template)}")
    if len(merged) != 201_248:
        raise ValueError("The merge changed the total number of scored rows")
    return merged


def save_scores(frame, output_dir, source_hash):
    csv_path = output_dir / "qwen_stage_a_pooled_scores.csv"
    frame.sort_values(["model", "dataset", "template_id", "row_index"]).to_csv(
        csv_path, index=False
    )
    db_path = output_dir / "qwen_stage_a_pooled_scores.sqlite3"
    temporary = db_path.with_suffix(".sqlite3.tmp")
    if temporary.exists():
        temporary.unlink()
    conn = sqlite3.connect(temporary)
    try:
        frame.to_sql("scores", conn, index=False, chunksize=5000)
        conn.execute("""
            CREATE UNIQUE INDEX score_key ON scores
            (model, dataset, split, template_id, row_index)
        """)
        conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO metadata VALUES ('source_sample_sha256', ?)", (source_hash,))
        conn.commit()
    finally:
        conn.close()
    temporary.replace(db_path)


def plot_overview(source_root, combined, frame, output_dir):
    tweet = load_from_disk(str(source_root / "artifacts" / "datasets" / "tweet_eval_sentiment"))["test"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    rows = (
        ("sst2", "combined", "SST-2 combined (n=12,872)", combined["text"]),
        ("tweet_eval_sentiment", "test", "TweetEval sentiment test (n=12,284)", tweet["text"]),
    )
    for y, (dataset, split, title, texts) in enumerate(rows):
        part = frame[(frame.dataset == dataset) & (frame.split == split)
                     & (frame.model == "0.5b") & (frame.template_id == "template_1")]
        counts = part.gold_label.value_counts().reindex(charts.LABELS[dataset], fill_value=0)
        axes[y, 0].bar(counts.index, counts.values,
                       color=[charts.COLORS[label] for label in counts.index])
        axes[y, 0].set_title(f"{title} · class counts")
        axes[y, 0].set_ylabel("Examples")
        for x, value in enumerate(counts.values):
            axes[y, 0].text(x, value, f"{value:,}", ha="center", va="bottom")
        axes[y, 1].hist([len(text.split()) for text in texts], bins=40,
                         color="#56b4e9", edgecolor="white")
        axes[y, 1].set_title(f"{title} · text length")
        axes[y, 1].set_xlabel("Words per example")
        axes[y, 1].set_ylabel("Examples")
    charts.save_figure(fig, output_dir, "dataset_overview_pooled")


def make_figures(source_root, combined, frame, metrics, output_dir):
    charts.DATASETS = (
        ("sst2", "combined", "SST-2 combined (n=12,872)"),
        ("tweet_eval_sentiment", "test", "TweetEval sentiment test (n=12,284)"),
    )
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    plot_overview(source_root, combined, frame, output_dir)
    charts.plot_performance(metrics, output_dir)
    charts.plot_confusions(frame, output_dir)
    charts.plot_margins(frame, output_dir)
    charts.plot_mean_margin_heatmap(frame, output_dir)
    charts.plot_prompt_disagreement(frame, output_dir)


def write_summary(frame, metrics, output_dir):
    lines = [
        "STAGE A - POOLED SST-2 AND TWEETEVAL RESULTS",
        "",
        "SST-2: 12,000 sampled train + 872 validation = 12,872 distinct rows.",
        "The original split and row index remain in the combined dataset and scores.",
        "TweetEval sentiment: 12,284 test rows. Four fixed prompts (T1-T4).",
        "SST-2 pooled is a project analysis set, not the official GLUE validation score.",
        "",
        "Each entry below is accuracy / macro-F1 / macro-recall, in percent.",
        "TweetEval's official sentiment metric is macro-recall.",
        "",
    ]
    for dataset, split, title in charts.DATASETS:
        lines.append(title)
        for model in charts.MODELS:
            values = []
            for template in charts.TEMPLATES:
                row = metrics[(metrics.dataset == dataset) & (metrics.split == split)
                              & (metrics.model == model) & (metrics.template_id == template)].iloc[0]
                values.append(
                    f"{template.replace('template_', 'T')} "
                    f"{row.accuracy*100:.2f}/{row.macro_f1*100:.2f}/{row.macro_recall*100:.2f}"
                )
            lines.append(f"  {model}: " + "  ".join(values))
        lines.append("")
    lines.extend([
        "M = log P(positive) - log P(negative) at the first answer token.",
        "Neutral has its own log-probability; M alone is not a 3-class classifier.",
        "Full scores: qwen_stage_a_pooled_scores.sqlite3 or .csv.",
        "Metric table: performance_metrics.csv. Figures: figures/ (PNG and PDF).",
    ])
    (output_dir / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    margin_summary = frame.groupby(
        ["model", "dataset", "split", "template_id", "gold_label"]
    )["margin"].agg(["count", "mean", "median", "std"]).reset_index()
    margin_summary.to_csv(output_dir / "margin_summary.csv", index=False)
    metrics.to_csv(output_dir / "performance_metrics.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "pooled_stage_a")
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    combined, train_indices, source_hash = make_combined_dataset(source_root, output_dir)
    frame = merge_scores(source_root, train_indices)
    metrics = charts.summarize_metrics(frame)
    save_scores(frame, output_dir, source_hash)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    make_figures(source_root, combined, frame, metrics, figures_dir)
    write_summary(frame, metrics, output_dir)
    print(f"Pooled SST-2 rows: {len(combined)}; all Qwen scores: {len(frame)}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
