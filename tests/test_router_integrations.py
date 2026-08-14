from __future__ import annotations

import base64
import inspect
import json
from io import BytesIO

from PIL import Image
from pydantic import BaseModel

from aic_pipeline import gemini_online, router_chat
from aic_pipeline.gemini_online import GeminiOnlineClient, GeminiResponse
from aic_pipeline.query_transformer import GeminiQueryTransformer
from aic_pipeline.router_batch import run_router_batch
from aic_pipeline.router_chat import RouterChatClient, RouterResponse
from aic_pipeline.qa_gemini import GeminiQAEngine
from scripts.run_gemini_scene_batch import scene_command


class FakeHttpResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class AnswerPayload(BaseModel):
    answer: str


class FakeRouter:
    model = "ag/gemini-3-flash"

    def __init__(self) -> None:
        self.calls = 0

    def json_schema_response(self, prompt: str, schema: dict, image: Image.Image | None,
                             *, temperature: float, max_tokens: int) -> RouterResponse:
        self.calls += 1
        assert prompt == "describe"
        assert schema["required"] == ["answer"]
        assert image is not None and image.size == (2, 2)
        assert temperature == 0
        assert max_tokens == 256
        return RouterResponse(
            text=json.dumps({"answer": "ok"}),
            usage={"total_tokens": 12},
            raw={"id": "response-1", "model": "gemini-3-flash",
                 "choices": [{"finish_reason": "stop"}]},
            latency_seconds=0.25,
        )


def test_router_batch_preserves_parser_envelope_and_resumes(tmp_path) -> None:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    request = {
        "key": "scene-1",
        "request": {
            "contents": [{"role": "user", "parts": [
                {"text": "describe"},
                {"inlineData": {"mimeType": "image/jpeg", "data": encoded}},
            ]}],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 256,
                "responseJsonSchema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        },
    }
    input_path = tmp_path / "requests.jsonl"
    output_path = tmp_path / "responses.jsonl"
    input_path.write_text(json.dumps(request) + "\n", encoding="utf-8")
    router = FakeRouter()

    first = run_router_batch(tmp_path, input_path, output_path, workers=1, client=router)
    second = run_router_batch(tmp_path, input_path, output_path, workers=1, client=router)

    envelope = json.loads(output_path.read_text(encoding="utf-8"))
    text = envelope["response"]["candidates"][0]["content"]["parts"][0]["text"]
    assert json.loads(text) == {"answer": "ok"}
    assert envelope["router"]["provider"] == "9router"
    assert envelope["router"]["latency_seconds"] == 0.25
    assert first["succeeded"] == second["succeeded"] == 1
    assert first["total_tokens"] == second["total_tokens"] == 12
    assert router.calls == 1


def test_scene_batch_does_not_override_router_model() -> None:
    command = scene_command("L21_V001", 7, 6)
    assert "--model" not in command


def test_router_uses_configured_64000_output_tokens(monkeypatch, tmp_path) -> None:
    requests = []

    def fake_urlopen(request, timeout: float):
        requests.append((request, timeout))
        return FakeHttpResponse({
            "choices": [{
                "message": {"content": '{"answer":"ok"}'},
                "finish_reason": "stop",
            }],
            "usage": {"total_tokens": 4},
        })

    monkeypatch.setenv("NINEROUTER_MAX_OUTPUT_TOKENS", "64000")
    monkeypatch.setattr(router_chat, "urlopen", fake_urlopen)
    client = RouterChatClient(tmp_path)
    client.json_schema_response(
        "answer", {"type": "object"}, temperature=0, max_tokens=1
    )

    payload = json.loads(requests[0][0].data)
    assert payload["max_tokens"] == 64000


def test_router_sends_qa_system_prompt(monkeypatch, tmp_path) -> None:
    requests = []

    def fake_urlopen(request, timeout: float):
        requests.append(request)
        return FakeHttpResponse({
            "choices": [{
                "message": {"content": '{"answer":"ok"}'},
                "finish_reason": "stop",
            }],
        })

    monkeypatch.setattr(router_chat, "urlopen", fake_urlopen)
    client = RouterChatClient(tmp_path)
    client.json_schema_response(
        "answer", {"type": "object"}, system_prompt="Ignore transcripts.",
        video_url="https://youtube.com/watch?v=example",
    )

    payload = json.loads(requests[0].data)
    assert payload["messages"][0] == {
        "role": "system", "content": "Ignore transcripts."
    }
    assert payload["messages"][1]["role"] == "user"
    assert payload["messages"][1]["content"][1] == {
        "type": "video_url",
        "video_url": {"url": "https://youtube.com/watch?v=example"},
    }


def test_online_gemini_uses_direct_api_key_and_structured_multimodal_request(
        monkeypatch, tmp_path) -> None:
    requests = []

    def fake_urlopen(request, timeout: float):
        requests.append((request, timeout))
        return FakeHttpResponse({
            "candidates": [{"content": {"parts": [{"text": '{"answer":"ok"}'}]}}],
            "usageMetadata": {"totalTokenCount": 9},
        })

    monkeypatch.setenv("GEMINI_API_KEY", "query-only-secret")
    monkeypatch.setenv("GEMINI_API_BASE_URL", "https://gemini.example/v1beta")
    monkeypatch.setattr(gemini_online, "urlopen", fake_urlopen)
    client = GeminiOnlineClient(tmp_path, model="gemini-test", timeout_seconds=17)
    result = client.json_response(
        "answer", AnswerPayload, Image.new("RGB", (2, 2), "blue"),
        system_prompt="Use visual evidence only.",
        video_url="https://youtube.com/watch?v=example",
        temperature=0,
        max_tokens=128,
    )

    request, timeout = requests[0]
    payload = json.loads(request.data)
    assert request.full_url.endswith("/models/gemini-test:generateContent")
    assert request.get_header("X-goog-api-key") == "query-only-secret"
    assert timeout == 17
    assert payload["systemInstruction"]["parts"][0]["text"] == "Use visual evidence only."
    assert payload["contents"][0]["parts"][1]["fileData"]["fileUri"].startswith(
        "https://youtube.com/"
    )
    assert payload["contents"][0]["parts"][2]["inlineData"]["mimeType"] == "image/jpeg"
    assert payload["generationConfig"]["responseJsonSchema"]["required"] == ["answer"]
    assert json.loads(result.text) == {"answer": "ok"}


def test_query_transformer_is_available_only_with_online_gemini_key(
        monkeypatch, tmp_path) -> None:
    index = tmp_path / "index"
    index.mkdir()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert not GeminiQueryTransformer(tmp_path, index).available
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    assert GeminiQueryTransformer(tmp_path, index).available


def test_qa_default_timeout_is_30_seconds() -> None:
    default = inspect.signature(GeminiQAEngine).parameters["request_timeout_ms"].default
    assert default == 30_000


def test_qa_reads_youtube_url_from_archive_metadata(tmp_path) -> None:
    metadata_dir = tmp_path / "archive" / "media-info"
    metadata_dir.mkdir(parents=True)
    metadata_dir.joinpath("L21_V015.json").write_text(
        json.dumps({"watch_url": "https://youtube.com/watch?v=4MADNqa2Ihw"}),
        encoding="utf-8",
    )
    engine = GeminiQAEngine.__new__(GeminiQAEngine)
    engine.root = tmp_path
    engine._video_url_cache = {}

    assert engine._video_url("L21_V015") == "https://youtube.com/watch?v=4MADNqa2Ihw"


def test_qa_retries_malformed_json_with_stop_finish_reason() -> None:
    class FlakyQAClient:
        model = "gemini-2.5-flash"

        def __init__(self) -> None:
            self.calls = 0
            self.video_urls = []

        def json_response(self, *args, **kwargs) -> GeminiResponse:
            self.calls += 1
            self.video_urls.append(kwargs.get("video_url"))
            if self.calls == 1:
                raise ValueError(
                    "9Router response does not contain a JSON object "
                    "(finish_reason=stop)"
                )
            return GeminiResponse(
                text=json.dumps({
                    "candidate_label": "C1",
                    "answer": "tay điều khiển",
                    "evidence_vi": "Vật thể hiện rõ trong khung hình.",
                    "confidence": 0.95,
                }),
                usage={"total_tokens": 10},
                raw={"candidates": [{"finishReason": "STOP"}]},
                latency_seconds=0.1,
            )

    candidate = {
        "video_id": "L21_V001",
        "scene_index": 0,
        "start_frame": 0,
        "end_frame_exclusive": 10,
        "best_frame_id": 5,
        "top_visual_frame_ids": [5],
        "representative_frame_ids": [5],
    }
    engine = GeminiQAEngine.__new__(GeminiQAEngine)
    engine.model = "gemini-2.5-flash"
    engine.client = FlakyQAClient()
    engine._grounder = False
    engine._sheet = lambda candidates: (Image.new("RGB", (2, 2)), {"C1": candidates[0]})
    engine._video_url = lambda video_id: "https://youtube.com/watch?v=example"

    result = engine.answer("Người đàn ông đang cầm gì?", [candidate], maximum_candidates=1)

    assert engine.client.calls == 2
    assert engine.client.video_urls == [
        "https://youtube.com/watch?v=example",
        "https://youtube.com/watch?v=example",
    ]
    assert result["answer"] == "tay điều khiển"
