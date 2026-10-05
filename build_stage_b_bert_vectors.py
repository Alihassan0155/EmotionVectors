"""B3-B6: checkpointed NNsight emotion directions for BERT-Large-Uncased.

Uses the same strict single-label GoEmotions B2 training rows, text-token
pooling, class means, and one-versus-other-seven contrast as the Qwen runner.
Neutral-PCA denoising remains pending B7-B8.
"""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import nnsight
import numpy as np
import torch
import transformers
from nnsight import NNsight
from transformers import AutoModel, AutoTokenizer

from build_stage_b_vectors import checkpoint, export_vectors, load_training


MODEL_ID = "google-bert/bert-large-uncased"
MAX_TOKENS = 512


def tokenized_batch(tokenizer, batch):
    encoded = tokenizer([row["text"] for row in batch], add_special_tokens=True,
                        padding=True, truncation=False, return_special_tokens_mask=True,
                        return_tensors="pt")
    if encoded["input_ids"].shape[1] > MAX_TOKENS:
        raise ValueError("An example exceeds BERT's 512-token limit; no truncation is allowed")
    text_mask = encoded["attention_mask"].bool() & ~encoded["special_tokens_mask"].bool()
    lengths = text_mask.sum(dim=1)
    if (lengths == 0).any().item():
        raise ValueError("An example has zero valid text tokens")
    model_inputs = {key: value.to("cuda") for key, value in encoded.items()
                    if key in ("input_ids", "attention_mask", "token_type_ids")}
    return model_inputs, text_mask.to("cuda"), lengths.tolist()


def extract_pooled(wrapped, inputs, text_mask, layers):
    mask = text_mask.unsqueeze(-1)
    denominator = mask.sum(dim=1)
    with torch.inference_mode():
        with wrapped.trace(inputs["input_ids"], attention_mask=inputs["attention_mask"],
                           token_type_ids=inputs.get("token_type_ids")):
            pooled_layers = []
            for layer_index in range(layers):
                # BertLayer.forward returns a tuple; item 0 is its residual output.
                residual = wrapped.encoder.layer[layer_index].nns_output[0]
                pooled_layers.append((residual.float() * mask).sum(dim=1) / denominator)
            pooled = torch.stack(pooled_layers, dim=1).cpu().save()
    if pooled.ndim != 3 or pooled.shape[1] != layers or not torch.isfinite(pooled).all().item():
        raise ValueError("Invalid pooled BERT activation shape or values")
    return pooled.numpy().astype(np.float16)


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(f"CUDA: {torch.cuda.get_device_name(0)}", flush=True)
    data, source_hash, b2_hash = load_training()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    base = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float16,
                                     local_files_only=True).eval().to("cuda")
    wrapped = NNsight(base)
    layers = len(base.encoder.layer)
    width = base.config.hidden_size
    signature = {"model_id": MODEL_ID, "source_sha256": source_hash,
                 "b2_vector_train_sha256": b2_hash,
                 "layers": layers, "hidden_size": width, "examples": len(data),
                 "activation_site": "BertLayer.forward tuple[0], after each encoder block",
                 "text_token_rule": "raw text with required [CLS]/[SEP]; exclude added special and padding tokens",
                 "pooling": "mean over valid WordPiece text-token encoder outputs",
                 "max_input_tokens": MAX_TOKENS, "activation_storage_dtype": "float16",
                 "model_forward_dtype": "float16", "nnsight_version": nnsight.__version__,
                 "torch_version": torch.__version__, "transformers_version": transformers.__version__}
    output = args.output.resolve()
    conn, pooled = checkpoint(output, signature, len(data), layers, width)
    try:
        completed = {row[0] for row in conn.execute("SELECT row_index FROM completed")}
        selected = range(min(args.limit, len(data))) if args.limit else range(len(data))
        pending = [index for index in selected if index not in completed]
        print(f"BERT: {len(completed):,} saved, {len(pending):,} pending; "
              f"{layers} layers x {width} hidden", flush=True)
        for start in range(0, len(pending), args.batch_size):
            indices = pending[start:start + args.batch_size]
            batch = [data[index] for index in indices]
            inputs, mask, lengths = tokenized_batch(tokenizer, batch)
            values = extract_pooled(wrapped, inputs, mask, layers)
            pooled[indices] = values
            pooled.flush()
            records = [(index, row["emotion"], row["id"], token_count,
                        hashlib.sha256(row["text"].encode("utf-8")).hexdigest())
                       for index, row, token_count in zip(indices, batch, lengths)]
            with conn:
                conn.executemany("INSERT INTO completed VALUES (?,?,?,?,?)", records)
            if start % (args.batch_size * 25) == 0 or start + len(indices) == len(pending):
                print(f"  BERT: {len(completed)+start+len(indices):,}/{len(data):,} complete", flush=True)
        total = conn.execute("SELECT COUNT(*) FROM completed").fetchone()[0]
        if total == len(data):
            export_vectors(data, pooled, output, MODEL_ID, signature,
                           layer_axis="encoder_layer_1_based")
        else:
            print(f"BERT: partial checkpoint {total:,}/{len(data):,}; resume without --limit", flush=True)
    finally:
        conn.close()
        del wrapped, base, tokenizer, pooled
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Only first N rows for a smoke run; resume without it")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "stage_b_raw_vectors" / "bert_large_uncased")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch-size and limit must be positive")
    run(args)


if __name__ == "__main__":
    main()
