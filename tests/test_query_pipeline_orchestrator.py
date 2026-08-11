from pathlib import Path

from scripts.run_query_pipeline import (
    QUERY_STAGE_NAMES,
    build_query_stages,
    parse_args,
    render_query_report,
    stage_in_range,
)


def make_args(root: Path, *extra: str):
    return parse_args(
        [
            "--root",
            str(root),
            "--query",
            "nguoi dang noi chuyen",
            *extra,
        ]
    )


def test_query_stages_are_separate_from_offline_pipeline(tmp_path: Path) -> None:
    args = make_args(tmp_path, "--video-id", "L21_V001")
    stages = build_query_stages(args)

    assert tuple(stage.name for stage in stages) == QUERY_STAGE_NAMES
    query = stages[QUERY_STAGE_NAMES.index("hybrid_query")].command
    assert "scripts/query_hybrid.py" in query
    assert "nguoi dang noi chuyen" in query
    assert "L21_V001" in query


def test_local_serving_management_can_be_disabled(tmp_path: Path) -> None:
    args = make_args(tmp_path, "--no-start-serving", "--no-check-serving")
    by_name = {stage.name: stage for stage in build_query_stages(args)}

    assert not by_name["serving_up"].enabled
    assert not by_name["serving_check"].enabled
    assert by_name["hybrid_query"].enabled


def test_query_backend_options_reach_child_commands(tmp_path: Path) -> None:
    args = make_args(
        tmp_path,
        "--elasticsearch-url",
        "http://elastic.example:9200",
        "--milvus-uri",
        "http://milvus.example:19530",
        "--level",
        "L1",
        "--device",
        "cpu",
    )
    by_name = {stage.name: stage for stage in build_query_stages(args)}

    assert "http://elastic.example:9200" in by_name["serving_check"].command
    assert "http://milvus.example:19530" in by_name["serving_check"].command
    assert "L1" in by_name["hybrid_query"].command
    assert "cpu" in by_name["hybrid_query"].command


def test_query_stage_range_is_inclusive() -> None:
    assert stage_in_range("serving_check", "serving_check", "hybrid_query")
    assert stage_in_range("hybrid_query", "serving_check", "hybrid_query")
    assert not stage_in_range("serving_up", "serving_check", "hybrid_query")


def test_visual_report_exports_result_images(tmp_path: Path) -> None:
    source = tmp_path / "workspace/keyframe_images/L21_V001/frame.jpg"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"fake-jpeg")
    html_path = tmp_path / "workspace/reports/query_report.html"
    image_directory = tmp_path / "workspace/reports/query_assets"
    query_result = {
        "query": "nguoi dang noi chuyen",
        "video_id": "L21_V001",
        "results": [
            {
                "video_id": "L21_V001",
                "frame_id": 120,
                "shot_id": 2,
                "pts_ms": 4000,
                "image_path": "workspace/keyframe_images/L21_V001/frame.jpg",
                "rrf_score": 0.1,
                "evidence": [
                    {"source": "milvus_keyframe", "rank": 1, "score": 0.5}
                ],
                "frame": {"ocr_text": "news presenter"},
            }
        ],
    }

    outputs = render_query_report(
        query_result,
        root=tmp_path,
        html_path=html_path,
        image_directory=image_directory,
    )

    assert html_path.is_file()
    assert len(outputs["exported_images"]) == 1
    assert Path(outputs["exported_images"][0]).is_file()
    assert "report_image_path" in query_result["results"][0]
    assert "<img" in html_path.read_text(encoding="utf-8")
