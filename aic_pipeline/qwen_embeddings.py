from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


QWEN3_MODEL_DIR = "qwen3-embedding-0.6b"
QWEN3_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
QWEN3_QUERY_INSTRUCTION = (
    "Given a Vietnamese or English description of a video moment, retrieve scene "
    "documents whose visible content, actions, ASR, OCR, names, numbers, spatial "
    "relations, and event order match the query."
)


def last_token_pool(last_hidden_state: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
    if bool((attention_mask[:, -1] == 1).all()):
        return last_hidden_state[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    rows = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[rows, sequence_lengths]


def query_prompt(query: str, instruction: str = QWEN3_QUERY_INSTRUCTION) -> str:
    return f"Instruct: {instruction}\nQuery: {query.strip()}"


def load_qwen3(model_path: Path, device: torch.device,
               dtype: torch.dtype) -> tuple[AutoTokenizer, AutoModel]:
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(
            f"Qwen3 embedding model is missing at {model_path}. "
            f"Download {QWEN3_MODEL_ID} into that directory before building the index."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, padding_side="left"
    )
    model = AutoModel.from_pretrained(
        str(model_path), local_files_only=True, dtype=dtype
    ).to(device).eval()
    return tokenizer, model
