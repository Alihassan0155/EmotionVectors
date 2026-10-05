"""Freeze leakage-checked GoEmotions vector-training and validation splits."""

import argparse
import csv
import hashlib
import json
import os
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

from datasets import load_from_disk

from select_goemotions_b1 import TARGETS, digest_rows


KEY_RULE = "Unicode NFKC, casefold, then collapse all whitespace"
OUTPUTS = (
    "vector_train", "pca_neutral_train", "emotion_validation",
    "neutral_validation_reserve",
)


def text_key(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def split_digest(dataset):
    digest = hashlib.sha256()
    for row in dataset:
        record = (row["source_split"], row["source_row_index"], row["id"],
                  row["text"], row["emotion"], row["labels"])
        digest.update(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def select_without_keys(dataset, forbidden, pool_name, exclusions):
    indices = []
    for index, row in enumerate(dataset):
        if text_key(row["text"]) in forbidden:
            exclusions.append((pool_name, row["source_split"], row["source_row_index"],
                               row["id"], row["emotion"],
                               hashlib.sha256(text_key(row["text"]).encode("utf-8")).hexdigest()))
        else:
            indices.append(index)
    return dataset.select(indices)


def prepare(b1_root):
    b1_manifest = json.loads((b1_root / "manifest.json").read_text(encoding="utf-8"))
    single = load_from_disk(str(b1_root / "target_single_label"))
    neutral = load_from_disk(str(b1_root / "neutral_only"))
    for name, pool, field in (("single", single, "target_sha256"),
                              ("neutral", neutral, "neutral_sha256")):
        if digest_rows(pool) != b1_manifest[field]:
            raise ValueError(f"B1 {name} pool does not match manifest")
    train_ids = set(single["train"]["id"]) | set(neutral["train"]["id"])
    val_ids = set(single["validation"]["id"]) | set(neutral["validation"]["id"])
    if train_ids & val_ids:
        raise ValueError("Official train and validation selected comment IDs overlap")

    emotion_train_keys = {text_key(text) for text in single["train"]["text"]}
    all_train_keys = emotion_train_keys | {text_key(text) for text in neutral["train"]["text"]}
    excluded = []
    outputs = {
        "vector_train": single["train"],
        "pca_neutral_train": select_without_keys(
            neutral["train"], emotion_train_keys, "pca_neutral_train", excluded),
        "emotion_validation": select_without_keys(
            single["validation"], all_train_keys, "emotion_validation", excluded),
        "neutral_validation_reserve": select_without_keys(
            neutral["validation"], all_train_keys, "neutral_validation_reserve", excluded),
    }
    for name in ("emotion_validation", "neutral_validation_reserve"):
        if {text_key(text) for text in outputs[name]["text"]} & all_train_keys:
            raise ValueError(f"Training text leaked into {name}")
    if {text_key(text) for text in outputs["pca_neutral_train"]["text"]} & emotion_train_keys:
        raise ValueError("Emotion training text leaked into neutral PCA pool")
    return outputs, excluded, b1_manifest


def counts(dataset):
    result = Counter(dataset["emotion"])
    return {name: result[name] for name in TARGETS}


def save(output_root, outputs, excluded, b1_manifest):
    manifest = {
        "stage": "B2", "source": "GoEmotions simplified B1 selections",
        "b1_target_sha256": b1_manifest["target_sha256"],
        "b1_neutral_sha256": b1_manifest["neutral_sha256"],
        "text_key_rule": KEY_RULE,
        "policy": [
            "Build vectors only from official train examples",
            "Remove from validation any text matching an emotion or neutral training text",
            "Remove from neutral PCA training any text matching emotion training text",
            "Keep neutral validation separate; never use it for PCA fitting",
            "Do not access GoEmotions test, SST-2, or TweetEval in Stage B",
        ],
        "counts": {name: len(dataset) for name, dataset in outputs.items()},
        "emotion_counts": {name: counts(outputs[name]) for name in
                           ("vector_train", "emotion_validation")},
        "sha256": {name: split_digest(dataset) for name, dataset in outputs.items()},
        "excluded_memberships": dict(Counter(item[0] for item in excluded)),
        "stage_c_layer_selection": "validation only; pending",
        "goemotions_test_use": "untouched",
    }
    manifest_path = output_root / "manifest.json"
    if output_root.exists():
        if not manifest_path.is_file() or json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Existing B2 output differs from current input or policy")
        for name, dataset in outputs.items():
            stored = load_from_disk(str(output_root / name))
            if split_digest(stored) != manifest["sha256"][name]:
                raise ValueError(f"Saved {name} differs")
        print(f"Verified existing B2 outputs: {output_root}")
        return
    output_root.mkdir(parents=True)
    for name, dataset in outputs.items():
        dataset.save_to_disk(str(output_root / name))
    with (output_root / "excluded_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("pool", "source_split", "source_row_index", "comment_id", "emotion", "normalized_text_sha256"))
        writer.writerows(excluded)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = ["STAGE B2 - GOEMOTIONS TRAIN / HELD-OUT SPLITS", "",
             "Train builds vectors. Validation selects layers in Stage C.",
             "Test is untouched. No sentiment datasets are used.",
             f"Text overlap rule: {KEY_RULE}.", ""]
    for name in OUTPUTS:
        lines.append(f"{name}: {manifest['counts'][name]:,} rows")
    lines.extend(["", "Excluded memberships:"])
    for name, count in manifest["excluded_memberships"].items():
        lines.append(f"  {name}: {count:,}")
    lines.extend(["", "Class counts, hashes, and policy: manifest.json",
                  "Excluded membership provenance: excluded_rows.csv"])
    (output_root / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((output_root / "README.txt").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b1-root", type=Path, default=ROOT / "artifacts" / "goemotions_b1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "goemotions_b2")
    args = parser.parse_args()
    outputs, excluded, b1_manifest = prepare(args.b1_root.resolve())
    save(args.output.resolve(), outputs, excluded, b1_manifest)


if __name__ == "__main__":
    main()
