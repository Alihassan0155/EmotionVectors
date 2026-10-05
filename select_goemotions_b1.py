"""Select strict single-label target and neutral GoEmotions pools.

Only official train and validation rows are considered. The two selected pools
retain their source split; no Stage B2 train/held-out assignment is made here.
"""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

from datasets import DatasetDict, load_dataset, load_from_disk


SOURCE_ID = "google-research-datasets/go_emotions"
CONFIG = "simplified"
SPLITS = ("train", "validation")
TARGETS = ("joy", "love", "excitement", "gratitude", "sadness", "anger", "fear", "disgust")
NEUTRAL = "neutral"
POLICY = "exactly one original GoEmotions label, equal to the selected class"


def select_pool(dataset, split, allowed_ids, names):
    indices = []
    for index, row in enumerate(dataset):
        labels = row["labels"]
        if len(labels) == 1 and labels[0] in allowed_ids and row["text"].strip():
            indices.append(index)
    selected = dataset.select(indices)
    selected = selected.add_column("emotion", [names[dataset[i]["labels"][0]] for i in indices])
    selected = selected.add_column("source_split", [split] * len(indices))
    selected = selected.add_column("source_row_index", indices)
    return selected


def digest_rows(dataset_dict):
    digest = hashlib.sha256()
    for split in SPLITS:
        for row in dataset_dict[split]:
            record = (split, row["source_row_index"], row["id"], row["text"],
                      row["emotion"], row["labels"])
            digest.update(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def check_pool(target, neutral, label_ids):
    for split in SPLITS:
        target_rows, neutral_rows = target[split], neutral[split]
        if set(target_rows["source_row_index"]) & set(neutral_rows["source_row_index"]):
            raise ValueError(f"Target and neutral pools overlap in {split}")
        for row in target_rows:
            if len(row["labels"]) != 1 or row["emotion"] not in TARGETS or row["labels"][0] != label_ids[row["emotion"]]:
                raise ValueError("Invalid target selection")
        for row in neutral_rows:
            if row["labels"] != [label_ids[NEUTRAL]] or row["emotion"] != NEUTRAL:
                raise ValueError("Invalid neutral selection")
    all_ids = [row["id"] for pool in (target, neutral) for split in SPLITS for row in pool[split]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Selected comment IDs overlap across pools or source splits")


def make_manifest(source, target, neutral, label_ids):
    counts = {}
    for split in SPLITS:
        class_counts = Counter(target[split]["emotion"])
        counts[split] = {"source_rows": len(source[split]),
                         "target_total": len(target[split]),
                         "target_by_emotion": {name: class_counts[name] for name in TARGETS},
                         "neutral_only": len(neutral[split]),
                         "excluded": len(source[split]) - len(target[split]) - len(neutral[split])}
    target_texts = {split: set(target[split]["text"]) for split in SPLITS}
    neutral_texts = {split: set(neutral[split]["text"]) for split in SPLITS}
    overlap_checks = {
        "target_train_validation_exact_texts": len(target_texts["train"] & target_texts["validation"]),
        "neutral_train_validation_exact_texts": len(neutral_texts["train"] & neutral_texts["validation"]),
        "target_neutral_exact_texts": len(
            (target_texts["train"] | target_texts["validation"]) &
            (neutral_texts["train"] | neutral_texts["validation"])),
        "b2_action": "Remove exact-text overlap before using validation as held-out data",
    }
    return {"source_dataset": SOURCE_ID, "config": CONFIG,
            "source_url": "https://huggingface.co/datasets/google-research-datasets/go_emotions",
            "source_splits": list(SPLITS),
            "source_fingerprints": {split: source[split]._fingerprint for split in SPLITS},
            "target_emotions": list(TARGETS), "neutral_label": NEUTRAL,
            "label_ids": {name: label_ids[name] for name in (*TARGETS, NEUTRAL)},
            "selection_policy": POLICY,
            "neutral_use": "PCA denoising only; excluded from contrastive class means",
            "counts": counts,
            "overlap_checks": overlap_checks,
            "target_sha256": digest_rows(target), "neutral_sha256": digest_rows(neutral),
            "stage_b2_assignment": "pending; original train/validation splits preserved"}


def save_or_verify(output, target, neutral, manifest):
    manifest_path = output / "manifest.json"
    if output.exists():
        if not manifest_path.is_file():
            raise FileExistsError(f"Output exists without a manifest: {output}")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in manifest.items()
               if key != "overlap_checks"):
            raise ValueError("Existing B1 selection differs from current source or policy")
        for name, expected, digest_key in (("target_single_label", target, "target_sha256"),
                                           ("neutral_only", neutral, "neutral_sha256")):
            stored = load_from_disk(str(output / name))
            if any(len(stored[split]) != len(expected[split]) for split in SPLITS) or digest_rows(stored) != manifest[digest_key]:
                raise ValueError(f"Saved {name} differs from selection")
        print(f"Existing selected datasets verified: {output}")
    else:
        output.mkdir(parents=True)
        target.save_to_disk(str(output / "target_single_label"))
        neutral.save_to_disk(str(output / "neutral_only"))
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["STAGE B1 — GOEMOTIONS SELECTION", "",
             "Single-label target classes: " + ", ".join(TARGETS),
             "Neutral-only examples are separate and reserved for PCA denoising.",
             "Only official train and validation splits are selected; test is untouched.",
             "No SST-2 or TweetEval examples are used.", ""]
    for split in SPLITS:
        c = manifest["counts"][split]
        lines.append(f"{split}: source={c['source_rows']:,}; target={c['target_total']:,}; neutral={c['neutral_only']:,}; excluded={c['excluded']:,}")
        lines.append("  " + ", ".join(f"{name}={c['target_by_emotion'][name]:,}" for name in TARGETS))
    overlaps = manifest["overlap_checks"]
    lines.extend(["", "Exact-text overlaps to resolve in B2:",
                  f"  target train/validation: {overlaps['target_train_validation_exact_texts']}",
                  f"  neutral train/validation: {overlaps['neutral_train_validation_exact_texts']}",
                  f"  target versus neutral: {overlaps['target_neutral_exact_texts']}",
                  ])
    lines.extend(["", "Selected datasets: target_single_label/ and neutral_only/",
                  "Provenance and checksums: manifest.json", "Stage B2 split assignment: pending"])
    (output / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved B1 selection: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "goemotions_b1")
    args = parser.parse_args()
    loaded = load_dataset(SOURCE_ID, CONFIG, split=list(SPLITS))
    source = dict(zip(SPLITS, loaded))
    label_names = source["train"].features["labels"].feature.names
    if source["validation"].features["labels"].feature.names != label_names:
        raise ValueError("GoEmotions label mapping differs across splits")
    names = {i: name for i, name in enumerate(label_names)}
    if not set((*TARGETS, NEUTRAL)).issubset(set(label_names)):
        raise ValueError("A requested emotion label is missing")
    label_ids = {name: label_names.index(name) for name in (*TARGETS, NEUTRAL)}
    target = DatasetDict({split: select_pool(source[split], split,
                         {label_ids[name] for name in TARGETS}, names) for split in SPLITS})
    neutral = DatasetDict({split: select_pool(source[split], split,
                          {label_ids[NEUTRAL]}, names) for split in SPLITS})
    check_pool(target, neutral, label_ids)
    manifest = make_manifest(source, target, neutral, label_ids)
    save_or_verify(args.output.resolve(), target, neutral, manifest)
    print((args.output.resolve() / "README.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
