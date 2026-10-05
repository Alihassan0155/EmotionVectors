# EmotionVectors

Research pipeline testing whether emotion directions learned from GoEmotions are present during sentiment classification. Models: Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, and BERT-large-uncased.

## Current progress

| Stage | Status | Main results |
| --- | --- | --- |
| A — sentiment baseline | Complete for all four models | `artifacts/stage_a_results.txt`; `artifacts/stage_a_four_models/` |
| B — emotion vectors | Complete for all four models | `artifacts/stage_b_raw_vectors/`; `artifacts/stage_b_final_vectors/` |
| C — emotion validation and layer selection | Complete for all four models | `artifacts/stage_c_goemotions/`; `artifacts/stage_c_empathetic_dialogues/` |
| D — observe emotion directions during sentiment classification | Planned; no Stage D results in this project yet | Next work below |

Stage A evaluates four fixed prompt templates (T1–T4) on 12,872 pooled SST-2 examples (12,000 training examples plus 872 validation examples) and 12,284 TweetEval test examples. It stores per-example predictions and positive-minus-negative log-probability margins, plus accuracy, macro-F1, macro-recall, and invalid-output rates. The pooled SST-2 score is a project analysis result, not an official validation score. Qwen and BERT use different scoring objectives; compare raw margins only within a model and prompt. See `artifacts/stage_a_results.txt` for results and prompt details.

Stage B uses 7,417 strict single-label GoEmotions training examples across joy, love, excitement, gratitude, sadness, anger, fear, and disgust. NNsight extracts every layer's residual activations; valid text tokens are mean-pooled, then class means are contrasted with the mean of the other seven classes. A separate neutral GoEmotions set supplies layer-specific PCA components explaining 50% of neutral variance; these components are projected out of the raw directions. Final vectors are in `artifacts/stage_b_final_vectors/<model>/emotion_directions_final.float32.npy`; the compact report is `artifacts/stage_b_final_vectors/stage_b_results.txt`.

Stage C validates raw and denoised vectors on 879 held-out GoEmotions examples and tests cross-dataset transfer on 490 unique EmpatheticDialogues validation situations (70 per emotion for seven directly mapped emotions; love excluded). Selected intervention layers:

| Model | Layer | GoEmotions / EmpatheticDialogues macro-AUROC |
| --- | ---: | ---: |
| Qwen 0.5B | 14 / 24 | 0.876 / 0.840 |
| Qwen 1.5B | 27 / 28 | 0.905 / 0.850 |
| Qwen 3B | 18 / 36 | 0.896 / 0.863 |
| BERT-large | 23 / 24 | 0.884 / 0.830 |

Selection used emotion-validation data only, never SST-2 or TweetEval. EmpatheticDialogues scores at selected layers are descriptive because that dataset helped select the layers. See `artifacts/stage_c_empathetic_dialogues/stage_c6_c7_results.txt` and `selected_intervention_layers.json` in the same folder.

## Next: Stage D

1. Freeze the exact sentiment examples and fixed Stage A prompt template(s) before analysis. Keep SST-2 train and validation provenance so results can be reported separately; do not tune layer choice using sentiment labels.
2. For each sentiment example, run each Qwen model with the chosen prompt and extract residual activations at every layer. Mean-pool only its text tokens using the Stage B/C masking rule. For BERT, use its matching Stage A cloze prompt and encoder activations, and report it separately from causal Qwen.
3. Calculate cosine similarity between each pooled activation and each of the eight **final denoised** emotion vectors. Save per-example scores with dataset, split, model, prompt, layer, gold sentiment, and stable example ID. Make extraction resumable.
4. Compare scores by gold sentiment class; report effect sizes and bootstrap 95% confidence intervals. Plot emotion-by-layer results, with TweetEval neutral shown separately. This stage observes representations only; steering and ablation come later.

Stage D has not been run in this checkout. Do not describe its planned outputs as findings.

## Moving to another device

Copy the source files, `.gitignore`, `README.md`, `requirements*.txt`, and any `artifacts/` results you need. **`artifacts/` is ignored by Git**, so transferring only a Git repository will omit the datasets, checkpoints, vectors, metrics, and reports. Copy `artifacts/` separately if you want to resume or audit completed runs. `visualizations/` is also ignored and can be copied separately if you need existing figures. The `.venv/` environment and `.cache/` model downloads can be recreated; they need not be transferred. This folder currently has no `.git` repository, so initialize Git on the source or destination if you want version control.

On Windows PowerShell, from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# Install the CUDA-enabled PyTorch build appropriate to the new GPU/driver first.
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-stage-b.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

Use the [PyTorch installation selector](https://pytorch.org/get-started/locally/) for the CUDA wheel matching the destination machine. Download Hugging Face model weights again if `.cache/` is not copied. Check each script's `--help` before resuming a stage. To reproduce UMAP figures, install `umap-learn` as an additional visualization dependency.

Older planning notes are in `STAGE_B_PLAN.md` and `CURRENT_STATE_2026-09-30.md`; this README and the output summaries above reflect the later completed stages.
