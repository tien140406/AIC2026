# AIC 2026 Video Preprocessing Pipeline

This repository contains the reproducible, per-video preprocessing pipeline
for AIC 2026. Generated artifacts, model checkpoints, the dataset, and Python
environments are intentionally kept out of Git.

The current implementation provides explicit CLI stages plus separate offline
and query orchestrators. There is no dataset-wide command that processes every
video end to end.

## Current pipeline

```text
archive/video/**/*.mp4
        |
        v
dataset audit -> workspace/manifests
        |
        v
OmniShotCut shot ranges
        |
        v
shot-local sampling (stride + optional anchors)
        |
        v
BEiT-3 visual embeddings (CUDA, L2 normalized)
        |
        v
adaptive spherical clustering -> medoids -> redundancy elimination
        |
        v
final keyframe manifest + compact embedding matrix
        |
        +--> OpenCLIP keyframe embeddings --------------------> Milvus keyframes
        +--> JPG -> PP-OCR -> Gemini OCR ---------------------> Elasticsearch
        +--> JPG -> Co-DETR metadata -------------------------> Elasticsearch
        |                 +--> ROI crop -> OpenCLIP ----------> Milvus ROIs
        +--> ASR/captions -> L0/L1 aligned segments ----------> Elasticsearch
                                                                  |
                         OpenCLIP text query -> Milvus ------------+
                                                                  v
                                                    RRF hybrid retrieval
```

Implemented stages:

- dataset audit and canonical `videos.parquet` / `provided_keyframes.parquet`
  manifests;
- GPU shot segmentation with the pinned OmniShotCut checkpoint;
- shot-local stride sampling with `frame_uid`, `pts_ms`, and anchor metadata;
- BEiT-3 vision features for adaptive keyframe selection;
- spherical clustering, medoid selection, protected anchors, and conservative
  temporal/shot redundancy removal;
- final-keyframe JPG export;
- OpenCLIP image embeddings for selected keyframes, or optionally dense sampled
  frames with `--sampling`;
- resumable ASR/caption extraction, L0/L1 chunks, keyframe alignment, and
  Elasticsearch/optional local SQLite FTS5 transcript search;
- resumable Gemini OCR with local PP-OCR screening;
- Co-DETR object inference, threshold/duplicate filtering, and OpenCLIP ROI
  embeddings;
- Elasticsearch frame/ASR indexing, separate Milvus keyframe/ROI collections,
  and hybrid Reciprocal Rank Fusion (RRF) retrieval;
- separate offline and query orchestrators: the offline path performs extraction,
  embedding, and indexing, while the query path only reads Elasticsearch and
  Milvus;
- a normalized SQLite catalog retained as a local compatibility/debug tool.

The following are intentionally not part of the current implementation:

- dense transcript/text embeddings and text-vector retrieval;
- learned multimodal reranking, AIC scorer, Q&A evidence aggregation, or a
  web/API UI;
- a dataset-wide orchestrator and completed generated outputs checked into the
  repository;
- wiring `models/beit3/beit3_base_itc_patch16_224.pth` into the extractor. The
  current selector uses the timm/Hugging Face model
  `hf_hub:timm/beit3_base_patch16_224.pt` as a vision-only backbone.

Gemini OCR and the first-time Co-DETR setup require network access. Inference
after the required assets are cached is local; ASR can be made fully local by
using `--skip-youtube`.

## Requirements and setup

- Linux or Windows with `uv`, Python 3.10 (the CUDA environments are pinned to
  it), FFmpeg on `PATH`, and a CUDA-capable GPU for OmniShotCut, BEiT-3,
  OpenCLIP, and Co-DETR;
- a checked-out AIC dataset under `archive/`;
- model checkpoints under `models/` (ignored by Git).

Initialize the OmniShotCut submodule and create only the environments needed
for the stages you intend to run:

```bash
git submodule update --init --recursive

uv sync --project environments/omnishotcut
uv sync --project environments/beit3-timm
uv sync --project environments/asr
uv sync --project environments/gemini       # optional OCR
uv sync --project environments/codetr       # optional objects
uv sync --project environments/serving      # Elasticsearch and Milvus indexing
```

Copy `.env.example` to `.env` when using Gemini or other configured providers.
For a reproducible local model cache, set:

```bash
export HF_HOME="$PWD/models/huggingface"
```

The first model load may download weights. Once the cache is complete, setting
`HF_HUB_OFFLINE=1` prevents Hugging Face access.

## Run the offline pipeline

Preview every command without running it:

```bash
python3 scripts/run_offline_pipeline.py \
  --video-id L21_V001 \
  --dry-run
```

Run extraction, embedding, OCR/ASR/object processing, then ingest the generated
documents and vectors into the local Elasticsearch and Milvus stack:

```bash
python3 scripts/run_offline_pipeline.py \
  --video-id L21_V001 \
  --recreate-serving
```

Existing deterministic artifacts are skipped by default. ASR, OCR, and object
stages keep their own checkpoints and resume incomplete work. Use `--force`
only when intentionally recomputing all selected stages; it also resubmits
Gemini OCR and can incur API usage.

Use `--from-stage` and `--to-stage` to run a subsection, or `--skip-asr`,
`--skip-ocr`, `--skip-objects`, and `--skip-serving` to disable optional
branches. `--list-stages` prints the accepted stage names. Each run writes its
state to `workspace/reports/offline_pipeline_<video_id>.json`. The compatibility
entry point `scripts/run_video_pipeline.py` runs the same offline pipeline.

## Run the query pipeline

The query orchestrator does not extract video artifacts or rebuild indices. By
default it starts the local Docker stack, checks Elasticsearch and Milvus, then
runs hybrid OpenCLIP/BM25 retrieval:

```bash
python3 scripts/run_query_pipeline.py \
  --query 'nguoi dang noi chuyen' \
  --video-id L21_V001 \
  --limit 20
```

When querying already-running local or production backends, skip infrastructure
management and provide connection settings through CLI arguments or environment
variables:

```bash
python3 scripts/run_query_pipeline.py \
  --query 'nguoi cam ro ca chua' \
  --no-start-serving \
  --no-check-serving \
  --elasticsearch-url "$ELASTICSEARCH_URL" \
  --milvus-uri "$MILVUS_URI"
```

Query credentials continue to come from `ELASTICSEARCH_API_KEY`,
`ELASTICSEARCH_USERNAME`, `ELASTICSEARCH_PASSWORD`, and `MILVUS_TOKEN`. Query
runs write execution reports to `workspace/reports/query_pipeline_<timestamp>.json`.

## Input layout

The audit stage scans `archive/video/**/*.mp4`. Other legacy/provided artifacts
are optional and are recorded in the audit report:

```text
archive/
  video/**/*.mp4
  media-info/<video_id>.json
  keyframes/<video_id>/*.jpg
  map-keyframes/<video_id>.csv
  clip-features-32/<video_id>.npy
  objects/<video_id>/*.json
  transcripts/<video_id>.json or .srt
```

`archive/keyframes`, `map-keyframes`, and `clip-features-32` are audit inputs;
the current keyframe selector consumes the generated BEiT-3 feature manifest
under `workspace/features/`.

## Run the visual keyframe pipeline

The commands below process one video. Replace `L21_V001` and the input path as
needed.

```bash
VIDEO_ID=L21_V001
VIDEO_PATH=archive/video/L21_V001.mp4
BEIT_ID=timm_beit3_base_patch16_224_pt

# Build/update workspace/manifests/videos.parquet.
uv run --project environments/omnishotcut \
  python scripts/audit_dataset.py

# Write workspace/shots/$VIDEO_ID.json.
uv run --project environments/omnishotcut \
  python scripts/run_omnishotcut.py \
  --input "$VIDEO_PATH"

# Write workspace/samples/$VIDEO_ID.parquet.
uv run --project environments/omnishotcut \
  python scripts/build_sampling_manifest.py \
  --video-id "$VIDEO_ID" \
  --stride 5

# Write workspace/features/$BEIT_ID/$VIDEO_ID.*.
uv run --project environments/beit3-timm \
  python scripts/extract_timm_beit3_embeddings.py \
  --video-id "$VIDEO_ID" \
  --batch-size 32

# Write workspace/keyframes/$BEIT_ID/$VIDEO_ID.parquet and the compact index.
uv run --project environments/beit3-timm \
  python scripts/select_keyframes.py \
  --video-id "$VIDEO_ID" \
  --encoder-id "$BEIT_ID"
```

`build_sampling_manifest.py --stride-only` omits shot start/middle/end and
boundary anchors. The selector defaults to protected core anchors, up to 12
clusters per shot, a cosine redundancy threshold of `0.985`, and a minimum of
3 retained frames per non-empty shot. All of these can be overridden in the
CLI; the selection report records the actual parameters.

## Retrieval embeddings and keyframe images

OpenCLIP runs after final keyframe selection. Its default model is
`ViT-B-16/laion2b_s34b_b88k`, cached under `models/openclip/`:

```bash
uv run --project environments/beit3-timm \
  python scripts/openclip_embeddings.py \
  --video-id "$VIDEO_ID" \
  --keyframe-encoder-id "$BEIT_ID" \
  --batch-size 32
```

Use `--sampling workspace/samples/$VIDEO_ID.parquet` instead of the selected
keyframe manifest only when intentionally building a dense frame index. The
extractor writes normalized FP16 vectors and a Parquet row mapping under
`workspace/features/<encoder_id>/`.

Export immutable JPGs for OCR and object detection:

```bash
uv run --project environments/beit3-timm \
  python scripts/export_keyframe_jpgs.py \
  --video-id "$VIDEO_ID" \
  --encoder-id "$BEIT_ID" \
  --quality 92
```

## ASR and transcript search

ASR is local and resumable when YouTube captions are disabled. The `auto`
provider checks existing local captions first, then optional YouTube captions,
then falls back to faster-whisper. See [ASR_PIPELINE.md](ASR_PIPELINE.md) for
provider and schema details.

```bash
uv run --project environments/asr \
  python scripts/extract_transcripts.py \
  --video-id "$VIDEO_ID" \
  --provider auto \
  --skip-youtube

uv run --project environments/asr \
  python scripts/build_transcript_index.py

uv run --project environments/asr \
  python scripts/query_transcript_index.py \
  --query 'ca chua' \
  --limit 10
```

Outputs include `workspace/transcripts/<video_id>/l0.parquet`, `l1.parquet`,
`manifest.json`, optional `workspace/audio/asr/<video_id>.wav`, and the FTS5
database at `workspace/indexes/transcripts.sqlite`. Elasticsearch indexing is
performed with the other serving artifacts after extraction finishes.

## OCR and objects (optional)

OCR uses local PP-OCRv5 screening and Gemini for structured frame OCR. It is
not an offline-only stage because the Gemini request requires
`GEMINI_API_KEY`:

```bash
uv run --project environments/gemini \
  python scripts/extract_gemini_ocr.py \
  --video-id "$VIDEO_ID" \
  --batch-size 6 \
  --concurrency 8 \
  --checkpoint-every 1000
```

Co-DETR setup downloads the pinned source/checkpoint once; inference and
filtering then run locally on the exported final-keyframe JPG manifest:

```bash
uv run --project environments/codetr python scripts/setup_codetr.py

uv run --project environments/codetr \
  python scripts/extract_codetr_objects.py \
  --video-id "$VIDEO_ID" \
  --score-threshold 0.30 \
  --checkpoint-every 100

uv run --project environments/codetr \
  python scripts/filter_object_detections.py \
  --output-dir workspace/objects/co_dino_5scale_lsj_r50_3x_coco/$VIDEO_ID \
  --score-threshold 0.40

uv run --project environments/beit3-timm \
  python scripts/openclip_roi_embeddings.py \
  --video-id "$VIDEO_ID" \
  --model-id co_dino_5scale_lsj_r50_3x_coco \
  --padding-ratio 0.05
```

OCR writes under `workspace/ocr/`; object inference writes raw and summary
tables under `workspace/objects/co_dino_5scale_lsj_r50_3x_coco/<video_id>/`.
OCR and object inference checkpoint per frame and can be resumed. ROI crops are
generated in memory; only their normalized FP16 OpenCLIP vectors and mapping
metadata are persisted under `workspace/features/`.

## Elasticsearch and Milvus serving

The production serving layout keeps OCR/object documents and ASR segments in
Elasticsearch. Keyframe and ROI OpenCLIP vectors are stored in separate Milvus
collections. Artifact files remain the rebuildable source of truth.

Start the production-like local stack before indexing. It runs Elasticsearch
and Milvus standalone (with etcd and MinIO) on loopback-only ports and keeps
their data in named Docker volumes:

```bash
docker compose -f environments/serving/docker-compose.local.yml up -d
docker compose -f environments/serving/docker-compose.local.yml ps

uv run --project environments/serving \
  python scripts/check_local_serving.py --smoke-write
```

The smoke check creates, searches, and removes one temporary Elasticsearch
index and one temporary Milvus collection. Stop the containers with:

```bash
docker compose -f environments/serving/docker-compose.local.yml down
```

`down` preserves the named volumes, so indexed data survives restarts. Do not
use `down -v` unless intentionally deleting all local Elasticsearch and Milvus
data.

Index frame metadata and ASR segments:

```bash
uv run --project environments/serving \
  python scripts/index_elasticsearch.py \
  --selection-encoder-id "$BEIT_ID" \
  --recreate

uv run --project environments/serving \
  python scripts/query_elasticsearch.py \
  --query 'ca chua person' \
  --limit 20
```

Ingest the OpenCLIP matrices directly from `workspace/features/`; this path
does not require the SQLite catalog:

```bash
uv run --project environments/serving \
  python scripts/ingest_milvus_artifacts.py \
  --entity-type keyframe \
  --encoder-id openclip_vit_b16_laion2b_s34b_b88k

uv run --project environments/serving \
  python scripts/ingest_milvus_artifacts.py \
  --entity-type roi \
  --encoder-id openclip_vit_b16_laion2b_s34b_b88k_roi
```

Run a text query through the query orchestrator, which invokes the OpenCLIP text
tower, both Milvus collections, and both Elasticsearch indices, then collapses
ROI/ASR evidence to `frame_uid` with RRF:

```bash
python3 scripts/run_query_pipeline.py \
  --query 'nguoi cam ro ca chua' \
  --no-start-serving \
  --limit 20
```

Keyframe vectors use `frame_uid` as the Milvus primary key. ROI vectors use
`roi_uid` and retain `parent_frame_uid`; ASR segments retain their aligned
`keyframe_uids`. See [SERVING_SCHEMA.md](SERVING_SCHEMA.md) for index mappings,
collection fields, and compatibility commands.

## Generated layout

```text
workspace/
  manifests/              # videos.parquet, provided_keyframes.parquet
  shots/                   # OmniShotCut JSON
  samples/                 # sampled frame manifests
  features/<encoder_id>/  # .f16.npy + Parquet row mappings
  keyframes/<encoder_id>/ # selected manifest, audit, compact vectors
  keyframe_images/<id>/   # exported JPGs + manifest.parquet
  transcripts/<id>/       # L0/L1 transcript artifacts
  ocr/                     # screening and Gemini OCR artifacts
  objects/<model>/<id>/   # raw detections and filtered tables
  indexes/                # transcript SQLite FTS5 index
  serving/                # optional local compatibility catalog.sqlite
  reports/                # per-stage JSON reports
```

## Development checks

The serving and transcript-index tests can be run without installing every
optional GPU/API environment:

```bash
PYTHONPATH=. uv run --project environments/serving \
  pytest -q tests/test_elasticsearch_milvus_artifacts.py \
    tests/test_serving_catalog.py tests/test_transcript_index.py \
    tests/test_video_pipeline_orchestrator.py \
    tests/test_query_pipeline_orchestrator.py
```

For the complete experimental roadmap and planned retrieval/fusion work, see
[PIPELINE_AIC2026.md](PIPELINE_AIC2026.md).
