# Query Serving Schema

The extraction pipeline keeps its immutable per-stage artifacts, then builds one
global SQLite catalog for metadata, reverse joins, sparse text search, and Milvus
provenance.

```text
pipeline artifacts
  -> workspace/serving/catalog.sqlite
       -> normalized metadata and reverse joins
       -> unified OCR + ASR FTS5
       -> Milvus keyframe collections
```

## Stable identity

Every frame-level artifact uses:

```text
frame_uid = <video_id>:<zero-padded-original-frame_id>
pts_ms    = integer presentation timestamp in milliseconds
```

`embedding_row` remains local to one matrix file and is never used as an
external primary key.

## Catalog tables

The catalog contains:

```text
videos
shots
keyframes
frame_embeddings
ocr_frames
ocr_blocks
object_frames
object_detections
asr_segments
segment_keyframes
text_documents
text_documents_fts
```

OCR blocks and object detections are normalized into one row per box. ASR
reverse lookup uses `segment_keyframes`; no serving query parses JSON arrays.
`text_documents_fts` indexes OCR and ASR in one FTS5 corpus while the source
tables retain full provenance.

## Build

Build the catalog only after final keyframe export and the desired metadata
stages have completed:

```bash
uv run --project environments/serving \
  python scripts/build_serving_catalog.py \
  --selection-encoder-id timm_beit3_base_patch16_224_pt
```

The build is atomic and writes:

```text
workspace/serving/catalog.sqlite
```

Rebuild after adding videos or rerunning an extraction stage. The artifact files
remain the source of truth.

## Query

Hydrate a Milvus result across all modalities:

```bash
uv run --project environments/serving \
  python scripts/query_serving_catalog.py \
  --frame-uid L21_V001:000000120
```

Search the unified OCR/ASR index:

```bash
uv run --project environments/serving \
  python scripts/query_serving_catalog.py \
  --query 'ca chua' --source-type asr --level L1
```

Filter detected objects through indexed normalized rows:

```bash
uv run --project environments/serving \
  python scripts/query_serving_catalog.py \
  --object-class person --min-score 0.7
```

## Milvus

Use a separate collection for each embedding model/version. The collection
primary key is `frame_uid`; scalar fields include video, frame, shot, timestamp,
text presence, and detected COCO class IDs.

```bash
uv run --project environments/serving \
  python scripts/ingest_milvus_keyframes.py \
  --model openclip_vit_b16_laion2b_s34b_b88k \
  --version openclip_vit_b16_laion2b_s34b_b88k \
  --uri http://localhost:19530
```

The ingestion uses cosine `AUTOINDEX`, batches upserts, and records the target
collection and indexing timestamp back into `frame_embeddings`.
