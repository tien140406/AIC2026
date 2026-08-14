from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aic_pipeline.qwen_embeddings import (
    QWEN3_MODEL_DIR,
    last_token_pool,
    load_qwen3,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build normalized local Qwen3 embeddings for scene documents."
    )
    parser.add_argument(
        "--documents", type=Path,
        default=ROOT / "workspace" / "offline_index" / "scenes.parquet",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "workspace" / "offline_index"
    )
    parser.add_argument(
        "--model-id", type=Path, default=ROOT / "models" / QWEN3_MODEL_DIR
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_length <= 0:
        raise ValueError("batch-size and max-length must be positive")

    table = pd.read_parquet(args.documents)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer, model = load_qwen3(args.model_id, device, dtype)

    def encode(texts: list[str]) -> np.ndarray:
        chunks = []
        for start in range(0, len(texts), args.batch_size):
            inputs = tokenizer(
                texts[start:start + args.batch_size], padding=True, truncation=True,
                max_length=args.max_length, return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                hidden = model(**inputs).last_hidden_state
                vectors = last_token_pool(hidden, inputs["attention_mask"]).float()
                vectors = torch.nn.functional.normalize(vectors, dim=-1)
            chunks.append(vectors.cpu().numpy())
        return np.concatenate(chunks).astype(np.float32)

    def join_values(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return " ".join(str(item) for item in value)
        except TypeError:
            return str(value)

    field_texts = {
        "retrieval": table.retrieval_text.fillna("").astype(str).tolist(),
        "summary": table.visual_summary.fillna("").astype(str).tolist(),
        "asr": (table.asr_text.fillna("").astype(str) + " "
                + table.asr_context.fillna("").astype(str)).tolist(),
        "ocr_entities": [
            f"{join_values(ocr)} {entities}".strip()
            for ocr, entities in zip(
                table.ocr_text, table.entities_text.fillna("")
            )
        ],
    }
    matrices = {}
    for field, texts in field_texts.items():
        print(json.dumps({"event": "encoding", "field": field, "rows": len(texts)}),
              flush=True)
        matrices[field] = encode(texts)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "text_embeddings_qwen3.npy", matrices["retrieval"])
    for field, matrix in matrices.items():
        if field != "retrieval":
            np.save(args.output / f"text_embeddings_qwen3_{field}.npy", matrix)
    table[["row_id", "document_id", "video_id", "scene_index"]].to_parquet(
        args.output / "text_embeddings_qwen3.parquet", index=False
    )
    report = {
        "model": str(args.model_id),
        "rows": len(table),
        "dimension": int(matrices["retrieval"].shape[1]),
        "fields": list(matrices),
        "device": str(device),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
    }
    (args.output / "qwen3_embedding_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
