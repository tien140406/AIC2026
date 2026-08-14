from __future__ import annotations

import torch

from aic_pipeline.qwen_embeddings import last_token_pool, query_prompt


def test_last_token_pool_handles_left_and_right_padding() -> None:
    hidden = torch.tensor([
        [[1.0], [2.0], [3.0]],
        [[4.0], [5.0], [6.0]],
    ])
    right_padded = torch.tensor([[1, 1, 0], [1, 1, 1]])
    assert last_token_pool(hidden, right_padded).tolist() == [[2.0], [6.0]]

    left_padded = torch.tensor([[0, 1, 1], [1, 1, 1]])
    assert last_token_pool(hidden, left_padded).tolist() == [[3.0], [6.0]]


def test_qwen_query_prompt_preserves_the_literal_query() -> None:
    value = query_prompt("áo số 84 ở bên trái")
    assert value.startswith("Instruct: ")
    assert value.endswith("Query: áo số 84 ở bên trái")
