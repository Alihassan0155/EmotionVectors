"""Freeze a balanced 490-conversation EmpatheticDialogues validation sample.

Reads only the official valid.csv archive member. Each example is the situation
prompt for one conversation; listener utterances do not inherit its label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

from prepare_goemotions_b2 import text_key


ROOT = Path(__file__).resolve().parent
OFFICIAL_URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
LABEL_MAP = {
    "joy": "joyful", "excitement": "excited", "gratitude": "grateful",
    "sadness": "sad", "anger": "angry", "fear": "afraid", "disgust": "disgusted",
}
PER_EMOTION = 70
SEED = "stage-c6-ed-validation-v1"


def load_official_validation(archive):
    by_id = {}
    with tarfile.open(archive, "r:gz") as handle:
        member = handle.extractfile("empatheticdialogues/valid.csv")
        if member is None:
            raise ValueError("Official archive has no valid.csv")
        reader = csv.DictReader(io.TextIOWrapper(member, encoding="utf-8", newline=""))
        for row in reader:
            conv_id = row["conv_id"]
            record = {"id": conv_id, "source_label": row["context"],
                      "text": row["prompt"].replace("_comma_", ",").strip()}
            if conv_id in by_id and by_id[conv_id] != record:
                raise ValueError(f"Inconsistent prompt or label for {conv_id}")
            by_id[conv_id] = record
    return list(by_id.values())


def goemotions_training_text_keys():
    folder = ROOT / "artifacts" / "goemotions_b2"
    forbidden = set()
    for name in ("vector_train", "pca_neutral_train"):
        data = load_from_disk(str(folder / name))
        forbidden.update(text_key(text) for text in data["text"])
    return forbidden


def freeze_sample(archive, output):
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    source = load_official_validation(archive)
    forbidden = goemotions_training_text_keys()
    excluded_overlap = 0
    seen_text = set()
    candidates = {emotion: [] for emotion in LABEL_MAP}
    reverse = {source_label: emotion for emotion, source_label in LABEL_MAP.items()}
    for row in sorted(source, key=lambda x: x["id"]):
        emotion = reverse.get(row["source_label"])
        if emotion is None:
            continue
        key = text_key(row["text"])
        if not key or key in forbidden or key in seen_text:
            excluded_overlap += 1
            continue
        seen_text.add(key)
        candidates[emotion].append({**row, "emotion": emotion})
    selected = []
    for emotion, rows in candidates.items():
        if len(rows) < PER_EMOTION:
            raise ValueError(f"Only {len(rows)} distinct {emotion} conversations; need {PER_EMOTION}")
        rows.sort(key=lambda row: hashlib.sha256(
            f"{SEED}|{row['id']}".encode("utf-8")).digest())
        selected.extend(rows[:PER_EMOTION])
    selected.sort(key=lambda row: hashlib.sha256(
        f"{SEED}|final|{row['id']}".encode("utf-8")).digest())
    if len(selected) != 490 or len({row["id"] for row in selected}) != 490:
        raise ValueError("Sample size or conversation uniqueness failed")
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "sample.jsonl"
    manifest_path = output / "sample_manifest.json"
    data_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected)
    manifest = {
        "source_url": OFFICIAL_URL, "source_archive_sha256": archive_hash,
        "source_member": "empatheticdialogues/valid.csv",
        "unit": "one unique situation prompt per conversation",
        "text_processing": "replace literal _comma_ with comma; strip outer whitespace",
        "label_map_goemotions_to_ed": LABEL_MAP,
        "excluded_emotion": "love (no direct EmpatheticDialogues label)",
        "sample_rule": "70 per emotion; sort by SHA256(seed|conv_id)", "seed": SEED,
        "sample_size": len(selected),
        "counts": dict(Counter(row["emotion"] for row in selected)),
        "candidate_counts_after_dedup_and_overlap_filter":
            {emotion: len(rows) for emotion, rows in candidates.items()},
        "excluded_exact_text_overlap_or_duplicate": excluded_overlap,
        "goemotions_train_overlap_rule": "NFKC casefold whitespace-collapse against vector_train and pca_neutral_train",
        "sample_sha256": hashlib.sha256(data_text.encode("utf-8")).hexdigest(),
        "use": "external validation only; never vector construction or PCA fitting",
    }
    if data_path.exists() or manifest_path.exists():
        if (not data_path.exists() or not manifest_path.exists() or
                data_path.read_text(encoding="utf-8") != data_text or
                json.loads(manifest_path.read_text(encoding="utf-8")) != manifest):
            raise ValueError("Existing frozen sample differs from current source or policy")
    else:
        data_path.write_bytes(data_text.encode("utf-8"))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen {len(selected)} conversations in {data_path}")
    print(manifest["counts"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        default=ROOT / "artifacts" / "empathetic_dialogues_official.tar.gz")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "stage_c_empathetic_dialogues")
    args = parser.parse_args()
    freeze_sample(args.archive.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
