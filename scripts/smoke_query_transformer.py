from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic_pipeline.query_transformer import GeminiQueryTransformer
from aic_pipeline.query_understanding import build_query_plan


queries = [
    "Trong cảnh vận động viên xe đạp mang số áo 84, đồng hồ thời gian ở góc màn hình đang hiển thị bao nhiêu?",
    "Một người phụ nữ mặc áo xanh đứng cạnh xe máy trên đường phố",
    "Biển báo có dòng chữ ROAD CLOSED xuất hiện ở đâu?",
]
parser = argparse.ArgumentParser(description="Smoke-test online Gemini query expansion.")
parser.add_argument(
    "--index", type=Path, default=ROOT / "workspace/offline_index_first20_local"
)
args = parser.parse_args()
transformer = GeminiQueryTransformer(ROOT, args.index)
if not transformer.available:
    raise RuntimeError("GEMINI_API_KEY is required for online query expansion")
results, timings = [], []
for query in queries:
    started = time.perf_counter()
    results.append(transformer.enhance(build_query_plan(query)).to_dict())
    timings.append(round(time.perf_counter() - started, 3))
print(json.dumps({"timings": timings, "results": results}, ensure_ascii=False, indent=2))
