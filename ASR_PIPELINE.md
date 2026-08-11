# ASR / Transcript Pipeline

The ASR stage is an offline, manifest-driven extension of the existing
video -> shots -> keyframes pipeline. It processes videos sequentially so one
lazy-loaded faster-whisper model is reused across the batch.

```text
videos.parquet + media-info + shots + keyframe manifest
  -> local JSON/SRT transcript (if present)
  -> YouTube captions (optional)
  -> FFmpeg mono 16 kHz WAV + faster-whisper fallback
  -> quality gate and conservative normalization
  -> L0 timestamped segments
  -> deterministic L1 chunks (L0 is retained)
  -> interval alignment to shots and keyframe frame_ids
  -> Parquet + manifest fingerprint
  -> optional SQLite FTS5 sparse index
```

## Runtime setup

```bash
uv sync --project environments/asr
```

The default Whisper model is `large-v3`; use
`--device cpu --compute-type int8` on a CPU machine. FFmpeg must be on `PATH`.

## Run extraction

One video, with optional providers disabled:

```bash
uv run --project environments/asr python scripts/extract_transcripts.py \
  --video-id L21_V001 --provider auto --skip-youtube
```

Batch/resume (the model is reused across videos):

```bash
uv run --project environments/asr python scripts/extract_transcripts.py \
  --all --max-videos 50 --language vi --batch-size 4
```

Useful switches are `--skip-youtube`, `--skip-vad`, `--no-semantic-chunks`,
`--force`, `--force-audio`, and `--dry-run`.

## Artifacts and schema

For `video_id=V`, outputs are written atomically to:

```text
workspace/transcripts/V/l0.parquet
workspace/transcripts/V/l1.parquet
workspace/transcripts/V/manifest.json
workspace/audio/asr/V.wav
```

Both Parquet levels contain `segment_id`, `video_id`, `level`, `start`, `end`,
`language`, provenance (`source`, `source_type`, `provider`, `model`), immutable
`raw_text`, normalized `clean_text`, `shot_ids`, `primary_shot`, original
keyframe `frame_id` values in `keyframe_ids`, confidence, `chunk_id`, quality,
creation time, and `metadata_json`.

L0 preserves provider timestamps. L1 groups adjacent L0 segments by duration,
gap, punctuation and character limits; it never replaces L0.

## Provider fallback

`auto` tries an existing `archive/transcripts/V.json` or `.srt`, then a YouTube
caption when a `watch_url` yields a video ID, and finally local Whisper. Missing,
malformed, wrong-language, or low-quality captions are recorded in the manifest
and fall through to Whisper. YouTube access is an optimization, never a hard
dependency.

## Alignment and indexing

Shot ranges are interpreted as `[start_frame, end_frame)` and converted using
the existing `frame / fps` convention. Every overlapping shot is attached to a
segment; the greatest-overlap shot is primary (earliest wins ties). Keyframes
are linked using their original `frame_id`, not a newly assigned row number.

Build the optional sparse index after extraction:

```bash
uv run --project environments/asr python scripts/build_transcript_index.py
uv run --project environments/asr python scripts/build_transcript_index.py \
  --query 'cà chua' --limit 10
```

The index is SQLite FTS5 at `workspace/indexes/transcripts.sqlite`. Search hits
already carry `video_id`, timestamps, shot IDs, and keyframe IDs. Use
`reciprocal_rank_fusion` for combining sparse/dense result lists and
`expand_temporal_neighbors` for configurable neighboring shots; these retrieval
helpers are intentionally separate from extraction.

The index also materializes `segment_keyframes`, keyed by the stable
`<video_id>:<zero-padded-frame_id>` frame UID. This avoids parsing
`keyframe_ids_json` for reverse lookups from a visual result to its aligned ASR
segments:

```bash
uv run --project environments/asr python scripts/query_transcript_index.py \
  --keyframe-video-id L21_V001 --frame-id 120 --limit 10

uv run --project environments/asr python scripts/query_transcript_index.py \
  --frame-uid L21_V001:000000120 --level L0 --limit 10
```

Covering indexes support both direct `frame_uid` lookups from Milvus results and
composite `(video_id, frame_id)` lookups from pipeline metadata. Both avoid a
full transcript scan and use indexed lookup as the collection grows. Rebuilding
the SQLite index also rebuilds all reverse links atomically, while the query CLI
opens the completed index read-only and never triggers a rebuild.

For production serving, `scripts/index_elasticsearch.py` writes one document per
L0/L1 segment to `aic-asr-segments-v1`, including `keyframe_uids` for joining
Milvus keyframe/ROI results. The SQLite FTS5 index remains useful for local
offline search and tests but is not required by the Elasticsearch/Milvus path.

Dense transcript embeddings are deliberately deferred until the repository has
a shared text-embedding/vector-index abstraction; the extraction schema is
already suitable for adding that index without rerunning ASR.
