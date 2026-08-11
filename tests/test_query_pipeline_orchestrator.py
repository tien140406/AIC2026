from pathlib import Path

from scripts.run_query_pipeline import (
    QUERY_STAGE_NAMES,
    build_query_stages,
    parse_args,
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
