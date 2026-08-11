from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aic_pipeline.embeddings.openclip import (
    DEFAULT_ENCODER_ID,
    DEFAULT_MODEL_NAME,
    DEFAULT_PRETRAINED,
)
from aic_pipeline.serving.elasticsearch import (
    DEFAULT_ASR_INDEX,
    DEFAULT_ELASTICSEARCH_URL,
    DEFAULT_FRAME_INDEX,
    create_client,
)
from aic_pipeline.serving.hybrid import hybrid_search
from aic_pipeline.serving.milvus_artifacts import default_collection_name


def encode_query(
    query: str,
    *,
    model_name: str,
    pretrained: str,
    model_cache: Path,
    device: torch.device,
) -> np.ndarray:
    model_cache.mkdir(parents=True, exist_ok=True)
    model = open_clip.create_model(
        model_name,
        pretrained=pretrained,
        device=device,
        precision="fp16" if device.type == "cuda" else "fp32",
        cache_dir=str(model_cache),
    ).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    tokens = tokenizer([query]).to(device)
    with torch.inference_mode():
        vector = model.encode_text(tokens, normalize=True)
    return vector[0].float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid OpenCLIP Milvus and Elasticsearch retrieval with RRF."
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--level", choices=("L0", "L1"), default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--embedding-version", default=DEFAULT_ENCODER_ID)
    parser.add_argument(
        "--model-cache", type=Path, default=PROJECT_ROOT / "models" / "openclip"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--elasticsearch-url",
        default=os.getenv("ELASTICSEARCH_URL", DEFAULT_ELASTICSEARCH_URL),
    )
    parser.add_argument(
        "--frame-index",
        default=os.getenv("ELASTICSEARCH_FRAME_INDEX", DEFAULT_FRAME_INDEX),
    )
    parser.add_argument(
        "--asr-index",
        default=os.getenv("ELASTICSEARCH_ASR_INDEX", DEFAULT_ASR_INDEX),
    )
    parser.add_argument("--elasticsearch-api-key", default=os.getenv("ELASTICSEARCH_API_KEY"))
    parser.add_argument("--elasticsearch-username", default=os.getenv("ELASTICSEARCH_USERNAME"))
    parser.add_argument("--elasticsearch-password", default=os.getenv("ELASTICSEARCH_PASSWORD"))
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--milvus-uri", default=os.getenv("MILVUS_URI", "http://localhost:19530"))
    parser.add_argument("--milvus-token", default=os.getenv("MILVUS_TOKEN"))
    parser.add_argument("--keyframe-collection", default=None)
    parser.add_argument("--roi-collection", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu to use CPU")
    vector = encode_query(
        args.query,
        model_name=args.model_name,
        pretrained=args.pretrained,
        model_cache=args.model_cache,
        device=device,
    )
    elasticsearch_client = create_client(
        args.elasticsearch_url,
        api_key=args.elasticsearch_api_key,
        username=args.elasticsearch_username,
        password=args.elasticsearch_password,
        verify_certs=not args.insecure,
    )
    try:
        from pymilvus import MilvusClient
    except ImportError as error:
        raise RuntimeError("Hybrid query requires pymilvus") from error
    milvus_client = (
        MilvusClient(uri=args.milvus_uri, token=args.milvus_token)
        if args.milvus_token
        else MilvusClient(uri=args.milvus_uri)
    )
    keyframe_collection = args.keyframe_collection or default_collection_name(
        "keyframe", DEFAULT_ENCODER_ID, args.embedding_version
    )
    roi_collection = args.roi_collection or default_collection_name(
        "roi", DEFAULT_ENCODER_ID, args.embedding_version
    )
    result = hybrid_search(
        elasticsearch_client=elasticsearch_client,
        milvus_client=milvus_client,
        query=args.query,
        query_vector=vector,
        keyframe_collection=keyframe_collection,
        roi_collection=roi_collection,
        frame_index=args.frame_index,
        asr_index=args.asr_index,
        video_id=args.video_id,
        level=args.level,
        candidate_limit=args.candidate_limit,
        limit=args.limit,
        rrf_k=args.rrf_k,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
