# EmotionVectors handoff — 2026-09-30

## Current status

Stage A sentiment baseline was run for Qwen2.5-0.5B-Instruct,
Qwen2.5-1.5B-Instruct, and BERT-Large-Uncased on SST-2 and TweetEval with four
fixed prompts. The compact report is
`artifacts/stage_a_three_models/README.txt`; per-example results are in
`artifacts/stage_a_three_models/scores.sqlite3`. Stage A sentiment data is not
used to build Stage B vectors.

Stage B uses the GoEmotions simplified split, restricted to strict single-label
examples for eight emotions: joy, love, excitement, gratitude, sadness,
anger, fear, and disgust. B2 has 7,417 vector-training examples, 879 held-out
emotion-validation examples, 12,810 neutral PCA-training examples, and 1,586
neutral-validation examples. The GoEmotions test split remains untouched.
See `artifacts/goemotions_b2/manifest.json` for counts, exclusions, and hashes.

B3–B6 have **completed** for both existing Qwen models. Each model's SQLite
checkpoint records **7,417/7,417** completed examples, and its vector
metadata says `complete: true` (verified 2026-09-30). The script is
`build_stage_b_vectors.py`, using NNsight on CUDA. It reads raw text without
chat formatting, captures each decoder block's residual output, excludes
padding/special tokens, and mean-pools valid text tokens. For each layer it
forms eight class means and raw directions
`mu_emotion - mean(mu_other_seven_emotions)`. These are **raw**, not yet
neutral-PCA-denoised, directions.

Results for each model are under `artifacts/stage_b_raw_vectors/0.5b/` and
`artifacts/stage_b_raw_vectors/1.5b/`:

- `pooled.float16.npy`: example × decoder layer × hidden dimension;
  7,417 × 24 × 896 for 0.5B and 7,417 × 28 × 1,536 for 1.5B.
- `progress.sqlite3`: completed example indices and provenance.
- `class_means.float32.npy`: emotion × decoder layer × hidden dimension.
- `emotion_directions_raw.float32.npy`: same axes, raw contrasts.
- `vector_norms.csv` and `vectors_metadata.json`: audit information.

## Next work

1. Continue the Stage B work. The four-model Stage A report is complete;
   `artifacts/stage_a_results.txt` records the SST-2 T4 invalid-output finding.
2. Extend `build_stage_b_vectors.py` with a `3b` model choice. Use the same
   frozen B2 training rows, token-pooling rule, layer capture, and contrast
   formula. Write to a distinct `artifacts/stage_b_raw_vectors/3b/`
   checkpoint; do not overwrite the completed 0.5B/1.5B artifacts. Run a
   small CUDA smoke pass, then resume all 7,417 examples.
3. Implement B7–B8: extract pooled activations from the separate neutral
   training pool, fit PCA separately at each layer, retain components that
   explain at least 50% of neutral variance, and project them out of the raw
   emotion directions. Keep validation and test data out of PCA fitting.
4. Save final denoised vectors (B9). Use only held-out emotion-validation data
   for layer choice in Stage C (B10); keep sentiment data out of layer choice.

## Qwen 3B Stage A update

Qwen2.5-3B-Instruct weights are now cached under
`D:\EmotionVectors\.cache\huggingface`. `qwen3b_stage_a.py` runs the same
100,624 frozen example/prompt combinations as the existing Qwen baseline,
using greedy generation and candidate-label log-probabilities in one
checkpointed pass. The user completed **100,624/100,624** rows in
`artifacts/qwen3b_stage_a/scores.sqlite3`. The four-model aggregate report
is complete in `artifacts/stage_a_four_models/`: **402,496** scores, metrics,
margin summaries, and plots. `artifacts/stage_a_results.txt` is the compact
four-model summary and records the Qwen 3B SST-2 T4 invalid-output finding.
