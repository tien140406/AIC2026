# AIC 2026 Offline Keyframe Pipeline

This repository currently tracks the reproducible offline pipeline through
BEiT-3 adaptive keyframe selection:

```text
dataset audit
  -> OmniShotCut shot segmentation
  -> shot-local stride sampling
  -> pluggable visual encoder (BEiT-3 baseline)
  -> adaptive spherical clustering
  -> medoid selection
  -> local/global/adjacent-shot redundancy elimination
```

Dataset files, checkpoints, virtual environments, and generated artifacts are
excluded from Git. OpenCLIP retrieval embeddings, ASR extraction, and Gemini
OCR are available as separate offline stages; multimodal fusion remains outside
this milestone while it is being evaluated.

## Runtime environments

```powershell
uv sync --project environments\omnishotcut
uv sync --project environments\beit3-timm
uv sync --project environments\asr
uv sync --project environments\gemini
```

OmniShotCut is pinned as a Git submodule under `third_party/OmniShotCut`.

## Pipeline example

```powershell
uv run --project environments\omnishotcut python scripts\audit_dataset.py

uv run --project environments\omnishotcut python scripts\run_omnishotcut.py `
  --video-id L21_V001

uv run --project environments\omnishotcut python scripts\build_sampling_manifest.py `
  --video-id L21_V001 --stride 5 --stride-only

uv run --project environments\beit3-timm python scripts\extract_timm_beit3_embeddings.py `
  --video-id L21_V001 --batch-size 32

uv run --project environments\beit3-timm python scripts\openclip_embeddings.py `
  --video-id L21_V001 `
  --keyframe-encoder-id timm_beit3_base_patch16_224_pt `
  --batch-size 32

uv run --project environments\beit3-timm python scripts\select_keyframes.py `
  --video-id L21_V001 `
  --encoder-id timm_beit3_base_patch16_224_pt `
  --protected-anchor-policy none `
  --single-cluster-max-distance 0.01 `
  --redundancy-cosine 0.95 `
  --temporal-window-frames 15 `
  --global-within-shot-cosine 0.995 `
  --adjacent-shot-cosine 0.995
```

## Pluggable encoders

Clustering consumes a Parquet feature manifest identified by `--encoder-id`.
The included OpenCLIP baseline uses `ViT-B-16/laion2b_s34b_b88k`, stores its
weights under `models/openclip/`, and embeds the selected keyframes by default
for image-text retrieval. Pass `--sampling` only when intentionally building a
dense frame index. Another visual model can be added by producing the same
metadata and embedding columns, without changing
`aic_pipeline/keyframes/selection.py`.

## Python package layout

Reusable implementation is grouped by pipeline feature; files in `scripts/`
only parse CLI arguments and dispatch to these modules.

```text
aic_pipeline/
  common/       # paths and atomic persistence
  dataset/      # dataset audit and manifests
  shots/        # shot-boundary detection
  keyframes/    # sampling, selection, and JPG export
  embeddings/   # BEiT-3 and OpenCLIP backends
  ocr/          # Gemini OCR schema and resumable pipeline
  asr/          # transcript providers, alignment, and FTS index
  objects/      # object-detection integrations
```

## Local directory layout

```text
archive/       # input dataset; ignored
models/        # downloaded checkpoints; ignored
workspace/     # shots, samples, embeddings, keyframes, reports; ignored
```

See `PIPELINE_AIC2026.md` for the larger experimental roadmap.

## Gemini frame OCR pilot

Export selected keyframes before calling Gemini. Pass the encoder explicitly so
the JPG manifest matches the selected keyframe manifest.

```bash
uv run --project environments/beit3-timm \
  python scripts/export_keyframe_jpgs.py \
  --video-id L21_V001 \
  --encoder-id timm_beit3_base_patch16_224_pt \
  --max-images 200

uv sync --project environments/gemini

uv run --project environments/gemini \
  python scripts/extract_gemini_ocr.py \
  --video-id L21_V001 \
  --limit 200 \
  --concurrency 8 \
  --checkpoint-every 1000
```

The OCR runner sends original frame images rather than contact sheets, validates
structured JSON, hashes images in parallel, and keeps only `--concurrency`
requests in flight. The Google SDK's nested retries are disabled so the runner
can distinguish temporary rate limits from a daily quota exhaustion. Temporary
limits use the server retry delay; a daily quota exhaustion stops new work and
flushes all completed raw checkpoints immediately.

Successful and failed requests are written atomically per frame under
`workspace/ocr/`. The aggregate `frames.parquet`, `failures.parquet`, and report
are refreshed every `--checkpoint-every` processed frames (1,000 by default)
and once more on completion, interruption, or quota exhaustion. Re-running the
same command skips successes whose image hash, model, and prompt version match.

Exit codes are `0` for success, `2` for completed work with frame failures, `3`
for terminal quota exhaustion, and `130` for an operator interruption.

## ASR transcript extraction

See `ASR_PIPELINE.md` for transcript extraction, provider fallback, alignment,
resumption, and SQLite FTS5 indexing.
