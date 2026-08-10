import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
from google.genai import errors
from PIL import Image
from pydantic import ValidationError

from aic_pipeline.ocr.gemini import (
    PROMPT_VERSION,
    FrameOcr,
    FrameOcrBatch,
    classify_error,
    frame_record,
    normalize_search_text,
    reusable_success,
    write_summary_tables,
)
from aic_pipeline.ocr import pipeline as runner


def test_frame_ocr_requires_has_text_to_match_blocks() -> None:
    with pytest.raises(ValidationError):
        FrameOcr.model_validate(
            {
                "frame_id": 10,
                "has_text": False,
                "blocks": [
                    {
                        "text": "AIC 2026",
                        "bbox_2d": [10, 20, 30, 40],
                        "legibility": "clear",
                    }
                ],
                "uncertainties": [],
            }
        )


def test_frame_ocr_batch_rejects_duplicate_frame_ids() -> None:
    frame = {
        "frame_id": 10,
        "has_text": False,
        "blocks": [],
        "uncertainties": [],
    }
    with pytest.raises(ValidationError):
        FrameOcrBatch.model_validate({"frames": [frame, frame]})


def test_normalize_search_text_preserves_diacritics_and_collapses_space() -> None:
    decomposed = "Tha\u0300nh   pho\u0302\u0301\nHo\u0302\u0300 Chi\u0301 Minh"
    assert normalize_search_text(decomposed) == "Thành phố Hồ Chí Minh"


def test_frame_record_keeps_verbatim_and_normalized_text() -> None:
    payload = {
        "video_id": "L21_V001",
        "shot_id": 3,
        "frame_id": 90,
        "pts_time": 3.0,
        "image_path": "/tmp/frame.jpg",
        "image_sha256": "abc",
        "model": "gemini-test",
        "prompt_version": "prompt-v1",
        "attempts": 1,
        "elapsed_seconds": 0.5,
        "processed_at": "2026-08-09T00:00:00+00:00",
        "usage": None,
        "ocr": {
            "frame_id": 90,
            "has_text": True,
            "blocks": [
                {
                    "text": "  AIC 2026  ",
                    "bbox_2d": [10, 20, 30, 40],
                    "legibility": "clear",
                },
                {
                    "text": "Việt Nam",
                    "bbox_2d": [50, 60, 70, 80],
                    "legibility": "partial",
                },
            ],
            "uncertainties": [],
        },
    }

    record = frame_record(payload)

    assert record["raw_text"] == "  AIC 2026  \nViệt Nam"
    assert record["normalized_text"] == "AIC 2026 Việt Nam"
    assert json.loads(record["blocks_json"])[1]["legibility"] == "partial"


def test_reusable_success_checks_image_model_and_prompt(tmp_path) -> None:
    path = tmp_path / "frame.json"
    path.write_text(
        json.dumps(
            {
                "image_sha256": "hash-1",
                "model": "gemini-test",
                "prompt_version": PROMPT_VERSION,
                "ocr": {},
            }
        ),
        encoding="utf-8",
    )

    assert (
        reusable_success(
            path,
            frame_id=90,
            image_sha256="hash-1",
            model="gemini-test",
        )
        is False
    )


def test_reusable_success_validates_frame_and_schema(tmp_path) -> None:
    path = tmp_path / "frame.json"
    path.write_text(
        json.dumps(
            {
                "frame_id": 90,
                "image_sha256": "hash-1",
                "model": "gemini-test",
                "prompt_version": PROMPT_VERSION,
                "ocr": {
                    "frame_id": 90,
                    "has_text": False,
                    "blocks": [],
                    "uncertainties": [],
                },
            }
        ),
        encoding="utf-8",
    )

    assert reusable_success(
        path, frame_id=90, image_sha256="hash-1", model="gemini-test"
    )
    assert not reusable_success(
        path, frame_id=91, image_sha256="hash-1", model="gemini-test"
    )
    assert not reusable_success(
        path, frame_id=90, image_sha256="hash-2", model="gemini-test"
    )


def test_classify_daily_quota_as_terminal() -> None:
    error = errors.ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": "Quota exhausted",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {
                        "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                    },
                    {"retryDelay": "44.5s"},
                ],
            }
        },
    )

    disposition = classify_error(error)

    assert disposition.category == "daily_quota_exhausted"
    assert disposition.terminal
    assert not disposition.retryable
    assert disposition.retry_after_seconds == 44.5


def test_classify_short_rate_limit_as_retryable() -> None:
    error = errors.ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": "Rate limited",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"},
                    {"retryDelay": "2s"},
                ],
            }
        },
    )

    disposition = classify_error(error)

    assert disposition.category == "rate_limited"
    assert disposition.retryable
    assert not disposition.terminal
    assert disposition.retry_after_seconds == 2.0


def test_process_batch_stops_immediately_on_daily_quota(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (16, 16), "white").save(image_path)
    api_error = errors.ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": "Quota exhausted",
                "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}
                ],
            }
        },
    )

    class FakeModels:
        calls = 0

        def generate_content(self, **kwargs):
            self.calls += 1
            raise api_error

    fake_models = FakeModels()
    fake_client = type("FakeClient", (), {"models": fake_models})()
    monkeypatch.setattr(runner, "client_for_thread", lambda *args: fake_client)
    stop_event = threading.Event()
    row = {
        "video_id": "L21_V001",
        "shot_id": 0,
        "frame_id": 10,
        "pts_time": 0.3,
        "image_path": str(image_path),
        "image_sha256": "hash",
    }

    results = runner.process_batch(
        [row],
        api_key="test-key",
        model="gemini-test",
        output_dir=tmp_path / "output",
        max_attempts=5,
        retry_base_seconds=0,
        request_timeout_seconds=1,
        stop_event=stop_event,
    )

    assert results[0]["status"] == "quota_exhausted"
    assert results[0]["payload"]["attempts"] == 1
    assert fake_models.calls == 1
    assert stop_event.is_set()


def test_process_batch_maps_results_and_passes_original_images_with_hints(
    tmp_path, monkeypatch
) -> None:
    first_path = tmp_path / "frame_10.jpg"
    second_path = tmp_path / "frame_20.jpg"
    Image.new("RGB", (128, 72), "white").save(first_path)
    Image.new("RGB", (96, 54), "black").save(second_path)

    class FakeModels:
        contents = None

        def generate_content(self, **kwargs):
            self.contents = kwargs["contents"]
            return type(
                "FakeResponse",
                (),
                {
                    "text": json.dumps(
                        {
                            "frames": [
                                {
                                    "frame_id": 20,
                                    "has_text": False,
                                    "blocks": [],
                                    "uncertainties": [],
                                },
                                {
                                    "frame_id": 10,
                                    "has_text": True,
                                    "blocks": [
                                        {
                                            "text": "HTV9 06:35:33 HD",
                                            "bbox_2d": [30, 810, 150, 940],
                                            "legibility": "clear",
                                        },
                                        {
                                            "text": "Intel",
                                            "bbox_2d": [500, 200, 800, 700],
                                            "legibility": "clear",
                                        },
                                    ],
                                    "uncertainties": [],
                                },
                            ]
                        }
                    ),
                    "usage_metadata": None,
                },
            )()

    fake_models = FakeModels()
    fake_client = type("FakeClient", (), {"models": fake_models})()
    monkeypatch.setattr(runner, "client_for_thread", lambda *args: fake_client)
    rows = [
        {
            "video_id": "L21_V001",
            "shot_id": 0,
            "frame_id": 10,
            "pts_time": 0.3,
            "image_path": str(first_path),
            "image_sha256": "hash-10",
            "screening_signature": "screen-hash-10",
            "screening_boxes_json": json.dumps(
                [{"bbox": [500, 200, 800, 700], "score": 0.9}]
            ),
        },
        {
            "video_id": "L21_V001",
            "shot_id": 1,
            "frame_id": 20,
            "pts_time": 0.6,
            "image_path": str(second_path),
            "image_sha256": "hash-20",
            "screening_signature": "screen-hash-20",
            "screening_boxes_json": "[]",
        },
    ]

    results = runner.process_batch(
        rows,
        api_key="test-key",
        model="gemini-test",
        output_dir=tmp_path / "output",
        max_attempts=1,
        retry_base_seconds=0,
        request_timeout_seconds=1,
        stop_event=threading.Event(),
    )

    assert [result["frame_id"] for result in results] == [10, 20]
    assert all(result["status"] == "success" for result in results)
    assert len(results[0]["payload"]["ocr"]["blocks"]) == 2
    assert results[1]["payload"]["ocr"]["has_text"] is False
    assert results[0]["payload"]["screening_boxes"] == [
        {"bbox": [500, 200, 800, 700], "score": 0.9}
    ]
    assert results[0]["payload"]["request_id"] == results[1]["payload"]["request_id"]
    assert results[0]["payload"]["batch_size"] == 2
    assert "FRAME_ID=10" in fake_models.contents[1]
    assert fake_models.contents[2].size == (128, 72)
    assert "FRAME_ID=20" in fake_models.contents[3]
    assert fake_models.contents[4].size == (96, 54)


def test_submit_until_full_keeps_only_bounded_jobs_in_flight(monkeypatch) -> None:
    release = threading.Event()

    def fake_process(batch, **kwargs):
        release.wait(timeout=2)
        return [
            {"status": "success", "frame_id": row["frame_id"], "payload": {}}
            for row in batch
        ]

    monkeypatch.setattr(runner, "process_batch", fake_process)
    jobs = iter(
        [
            [{"frame_id": 0}, {"frame_id": 1}],
            [{"frame_id": 2}, {"frame_id": 3}],
            [{"frame_id": 4}],
        ]
    )
    in_flight = {}
    stop_event = threading.Event()
    executor = ThreadPoolExecutor(max_workers=3)
    try:
        exhausted = runner.submit_until_full(
            executor,
            jobs,
            in_flight,
            concurrency=3,
            submit_kwargs={},
            stop_event=stop_event,
        )

        assert not exhausted
        assert len(in_flight) == 3
        assert [[row["frame_id"] for row in batch] for batch in in_flight.values()] == [
            [0, 1],
            [2, 3],
            [4],
        ]
    finally:
        release.set()
        executor.shutdown(wait=True, cancel_futures=True)


def test_sum_usage_counts_each_batch_request_once() -> None:
    frames = pd.DataFrame(
        [
            {
                "frame_id": 10,
                "request_id": "request-a",
                "usage_json": json.dumps({"prompt_token_count": 100}),
            },
            {
                "frame_id": 20,
                "request_id": "request-a",
                "usage_json": json.dumps({"prompt_token_count": 100}),
            },
            {
                "frame_id": 30,
                "request_id": "request-b",
                "usage_json": json.dumps({"prompt_token_count": 40}),
            },
        ]
    )

    assert runner.sum_usage(frames) == {"prompt_token_count": 140}


def test_summary_tables_prefer_success_over_old_failure(tmp_path) -> None:
    output_dir = tmp_path / "ocr"
    raw_dir = output_dir / "raw"
    failure_dir = output_dir / "failures"
    raw_dir.mkdir(parents=True)
    failure_dir.mkdir(parents=True)
    success = {
        "video_id": "L21_V001",
        "shot_id": 0,
        "frame_id": 10,
        "pts_time": 0.3,
        "image_path": "/tmp/frame.jpg",
        "image_sha256": "hash",
        "model": "gemini-test",
        "prompt_version": PROMPT_VERSION,
        "attempts": 1,
        "elapsed_seconds": 0.5,
        "processed_at": "2026-08-09T00:00:00+00:00",
        "usage": None,
        "ocr": {
            "frame_id": 10,
            "has_text": False,
            "blocks": [],
            "uncertainties": [],
        },
    }
    old_failure = {
        "video_id": "L21_V001",
        "shot_id": 0,
        "frame_id": 10,
        "pts_time": 0.3,
        "image_path": "/tmp/frame.jpg",
        "image_sha256": "hash",
        "model": "gemini-test",
        "prompt_version": PROMPT_VERSION,
        "attempts": 1,
        "error_type": "ClientError",
        "error": "old error",
        "failed_at": "2026-08-09T00:00:00+00:00",
    }
    unresolved_failure = {**old_failure, "frame_id": 20}
    (raw_dir / "frame_000000010.json").write_text(json.dumps(success), encoding="utf-8")
    (failure_dir / "frame_000000010.json").write_text(
        json.dumps(old_failure), encoding="utf-8"
    )
    (failure_dir / "frame_000000020.json").write_text(
        json.dumps(unresolved_failure), encoding="utf-8"
    )

    frames, failures = write_summary_tables(output_dir)

    assert frames["frame_id"].tolist() == [10]
    assert failures["frame_id"].tolist() == [20]
    assert (output_dir / "frames.parquet").is_file()
    assert (output_dir / "failures.parquet").is_file()


def test_summary_tables_filter_stale_prompt_and_screening_results(tmp_path) -> None:
    output_dir = tmp_path / "ocr"
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True)
    base = {
        "video_id": "L21_V001",
        "shot_id": 0,
        "pts_time": 0.3,
        "image_path": "/tmp/frame.jpg",
        "image_sha256": "hash",
        "model": "gemini-test",
        "attempts": 1,
        "elapsed_seconds": 0.5,
        "processed_at": "2026-08-09T00:00:00+00:00",
        "usage": None,
        "ocr": {
            "frame_id": 10,
            "has_text": False,
            "blocks": [],
            "uncertainties": [],
        },
    }
    current = {
        **base,
        "frame_id": 10,
        "prompt_version": PROMPT_VERSION,
        "screening_signature": "current-screen",
    }
    stale = {
        **base,
        "frame_id": 20,
        "prompt_version": "old-prompt",
        "screening_signature": "old-screen",
        "ocr": {**base["ocr"], "frame_id": 20},
    }
    (raw_dir / "frame_000000010.json").write_text(
        json.dumps(current), encoding="utf-8"
    )
    (raw_dir / "frame_000000020.json").write_text(
        json.dumps(stale), encoding="utf-8"
    )

    frames, _ = write_summary_tables(
        output_dir,
        selected_signatures={10: "current-screen"},
        model="gemini-test",
        prompt_version=PROMPT_VERSION,
    )

    assert frames["frame_id"].tolist() == [10]


def test_main_checkpoints_every_configured_number_of_frames(monkeypatch) -> None:
    jobs = pd.DataFrame(
        [
            {
                "video_id": "L21_V001",
                "frame_id": frame_id,
                "image_path": f"/tmp/{frame_id}.jpg",
                "image_sha256": f"hash-{frame_id}",
            }
            for frame_id in range(5)
        ]
    )
    checkpoint_statuses = []

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(runner, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "load_jobs", lambda *args, **kwargs: jobs)
    monkeypatch.setattr(runner, "reusable_success", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        runner,
        "process_batch",
        lambda batch, **kwargs: [
            {
                "status": "success",
                "frame_id": row["frame_id"],
                "payload": {},
            }
            for row in batch
        ],
    )

    def fake_checkpoint_report(**kwargs):
        checkpoint_statuses.append(kwargs["status"])
        return {"selected_failures": 0}

    monkeypatch.setattr(runner, "checkpoint_report", fake_checkpoint_report)
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            "extract_gemini_ocr.py",
            "--video-id",
            "L21_V001",
            "--no-screening",
            "--batch-size",
            "2",
            "--concurrency",
            "1",
            "--checkpoint-every",
            "2",
            "--progress-every",
            "100",
        ],
    )

    runner.main()

    assert checkpoint_statuses == ["running", "running", "completed"]


def test_main_defaults_to_six_frames_per_gemini_request(monkeypatch) -> None:
    jobs = pd.DataFrame(
        [
            {
                "video_id": "L21_V001",
                "frame_id": frame_id,
                "image_path": f"/tmp/{frame_id}.jpg",
                "image_sha256": f"hash-{frame_id}",
            }
            for frame_id in range(7)
        ]
    )
    batch_sizes = []

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(runner, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "load_jobs", lambda *args, **kwargs: jobs)
    monkeypatch.setattr(runner, "reusable_success", lambda *args, **kwargs: False)

    def fake_process(batch, **kwargs):
        batch_sizes.append(len(batch))
        return [
            {"status": "success", "frame_id": row["frame_id"], "payload": {}}
            for row in batch
        ]

    monkeypatch.setattr(runner, "process_batch", fake_process)
    monkeypatch.setattr(
        runner,
        "checkpoint_report",
        lambda **kwargs: {"selected_failures": 0},
    )
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            "extract_gemini_ocr.py",
            "--video-id",
            "L21_V001",
            "--no-screening",
            "--concurrency",
            "1",
            "--progress-every",
            "100",
        ],
    )

    runner.main()

    assert batch_sizes == [6, 1]


def test_main_stops_queue_and_checkpoints_on_daily_quota(monkeypatch) -> None:
    jobs = pd.DataFrame(
        [
            {
                "video_id": "L21_V001",
                "frame_id": frame_id,
                "image_path": f"/tmp/{frame_id}.jpg",
                "image_sha256": f"hash-{frame_id}",
            }
            for frame_id in range(5)
        ]
    )
    processed_frames = []
    checkpoint_statuses = []

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(runner, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "load_jobs", lambda *args, **kwargs: jobs)
    monkeypatch.setattr(runner, "reusable_success", lambda *args, **kwargs: False)

    def fake_process(batch, **kwargs):
        processed_frames.extend(row["frame_id"] for row in batch)
        kwargs["stop_event"].set()
        return [
            {
                "status": "quota_exhausted",
                "frame_id": row["frame_id"],
                "payload": {"error_category": "daily_quota_exhausted"},
            }
            for row in batch
        ]

    def fake_checkpoint_report(**kwargs):
        checkpoint_statuses.append(kwargs["status"])
        return {"selected_failures": 1}

    monkeypatch.setattr(runner, "process_batch", fake_process)
    monkeypatch.setattr(runner, "checkpoint_report", fake_checkpoint_report)
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            "extract_gemini_ocr.py",
            "--video-id",
            "L21_V001",
            "--no-screening",
            "--batch-size",
            "2",
            "--concurrency",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        runner.main()

    assert exit_info.value.code == 3
    assert processed_frames == [0, 1]
    assert checkpoint_statuses == ["quota_exhausted"]


def test_main_checkpoints_on_keyboard_interrupt(monkeypatch) -> None:
    jobs = pd.DataFrame(
        [
            {
                "video_id": "L21_V001",
                "frame_id": 0,
                "image_path": "/tmp/0.jpg",
                "image_sha256": "hash-0",
            }
        ]
    )
    checkpoint_statuses = []

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(runner, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "load_jobs", lambda *args, **kwargs: jobs)
    monkeypatch.setattr(runner, "reusable_success", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        runner,
        "process_batch",
        lambda batch, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    def fake_checkpoint_report(**kwargs):
        checkpoint_statuses.append(kwargs["status"])
        return {"selected_failures": 0}

    monkeypatch.setattr(runner, "checkpoint_report", fake_checkpoint_report)
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            "extract_gemini_ocr.py",
            "--video-id",
            "L21_V001",
            "--no-screening",
            "--concurrency",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        runner.main()

    assert exit_info.value.code == 130
    assert checkpoint_statuses == ["interrupted"]
