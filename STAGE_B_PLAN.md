# Stage B — GoEmotions emotion vectors

**Status (2026-09-30):** B1–B2 data prepared; B3–B6 completed for Qwen2.5
0.5B and 1.5B (7,417/7,417 rows each); B7–B10 planned. A Qwen2.5 3B
extension is planned but has not been run. This adapts Anthropic-style vector
construction to GoEmotions rather than replicating Anthropic's synthetic-story
dataset.

## B1. Target classes

Use these eight GoEmotions classes: `joy`, `love`, `excitement`, `gratitude`,
`sadness`, `anger`, `fear`, and `disgust`. Use GoEmotions `neutral` examples only
for PCA denoising, not as the primary contrast class.

B1 selects comments with exactly one original label equal to a target emotion,
plus a separate neutral-only pool. The selected train/validation datasets and
counts are saved under `artifacts/goemotions_b1/`; see its `manifest.json` and
`README.txt`.

## B2. Data split

Create an emotion-vector training set for vector construction and a disjoint
held-out emotion set reserved for Stage C validation. Do not use SST-2 or
TweetEval at any point in Stage B.

B2 saves leakage-checked pools under `artifacts/goemotions_b2/`. It keeps
official training examples for vector construction, filters validation texts
that match any training text after Unicode NFKC normalization, casefolding,
and whitespace collapse, and excludes emotion-training text from the neutral
PCA pool. It reserves neutral validation separately and never fits PCA on it.
Original split and row provenance is kept. The official test split remains
unused.

## B3. Activations

Run every emotion-vector training example through each Qwen model. Use NNsight
to explicitly access residual-stream activations at every layer.

`build_stage_b_vectors.py` uses NNsight to capture the output of every Qwen
decoder block, separately for both Qwen models. It runs on CUDA and checkpoints
pooled per-example activations so interrupted runs can resume. The initial
NNsight extraction was checked against Hugging Face hidden states. Both
training passes are complete as of 2026-09-30.

## B4. Token pooling

For each example and layer, average the residual activations over **all valid
text tokens**, excluding special and padding tokens. Pre-register this pooling
rule. Do not apply a fixed token-50 cutoff: GoEmotions comments are often much
shorter than the paragraph-length stories for which that rule was designed.

## B5. Class means

At each layer `l`, average the pooled training-example activations for each
of the eight emotions, obtaining `mu[e,l]`.

The runner saves these class means as `class_means.float32.npy` after all
training rows for a model are complete.

## B6. Contrastive raw vectors

For every emotion `e` and layer `l`, construct:

```text
v_raw[e,l] = mu[e,l] - mean(mu[o,l] for o in other seven emotions)
```

The primary contrast is **one emotion versus the other seven emotion means**,
not emotion versus neutral.

The runner saves these undenoised directions as
`emotion_directions_raw.float32.npy`. They are not the final PCA-denoised
vectors.

## B7. Neutral activation set

Use a separate set of GoEmotions neutral examples. Run them through each Qwen
model and apply the same layerwise residual extraction and text-token pooling
to form a neutral activation matrix per layer.

## B8. Neutral-PCA denoising

For each layer, fit PCA to the neutral activation matrix. Retain the smallest
number of principal components explaining at least 50% of its variance.
Project the retained components out of each raw emotion vector at that layer.

## B9. Save final vectors

Save the eight denoised emotion vectors at **every residual-stream layer** for
both Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct, with the class means,
raw vectors, PCA components, and data/split metadata needed to reproduce them.
The output inventory is `8 emotions × N layers × 2 Qwen models`.

## B10. Layer selection boundary

Do not select a final layer in Stage B. Stage C will choose the layer using
only the held-out emotion-validation data, without consulting SST-2 or
TweetEval sentiment outcomes.
