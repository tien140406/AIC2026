from pathlib import Path

from scripts.run_video_pipeline import (
    STAGE_NAMES,
    Stage,
    build_stages,
    parse_args,
    stage_in_range,
    stage_is_cached,
)


def make_args(root: Path, *extra: str):
    return parse_args(
        ["--root", str(root), "--video-id", "L21_V001", *extra]
    )


def test_build_stages_covers_pipeline_in_order(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    stages = build_stages(args, tmp_path)

    assert tuple(stage.name for stage in stages) == STAGE_NAMES
    assert "--skip-youtube" in stages[STAGE_NAMES.index("asr")].command
    assert stages[STAGE_NAMES.index("serving_up")].command[-1] == "--wait"
    assert STAGE_NAMES[-1] == "milvus_rois"


def test_optional_stage_groups_can_be_disabled(tmp_path: Path) -> None:
    args = make_args(
        tmp_path,
        "--skip-asr",
        "--skip-ocr",
        "--skip-objects",
        "--skip-serving",
    )
    by_name = {stage.name: stage for stage in build_stages(args, tmp_path)}

    assert not by_name["asr"].enabled
    assert not by_name["ocr"].enabled
    assert not by_name["objects"].enabled
    assert not by_name["openclip_rois"].enabled
    assert not by_name["elasticsearch"].enabled
    assert not by_name["milvus_rois"].enabled


def test_cached_outputs_resume_unless_force_is_requested(tmp_path: Path) -> None:
    output = tmp_path / "artifact.parquet"
    output.write_text("ready", encoding="utf-8")
    stage = Stage("sampling", ("example",), cached_outputs=(output,))

    assert stage_is_cached(stage, force=False)
    assert not stage_is_cached(stage, force=True)
    assert stage_is_cached(
        Stage(
            "codetr_setup",
            ("example",),
            cached_outputs=(output,),
            rerun_on_force=False,
        ),
        force=True,
    )


def test_stage_range_is_inclusive() -> None:
    assert stage_in_range("sampling", "sampling", "keyframes")
    assert stage_in_range("keyframes", "sampling", "keyframes")
    assert not stage_in_range("shots", "sampling", "keyframes")
    assert not stage_in_range("openclip_keyframes", "sampling", "keyframes")


def test_force_and_serving_rebuild_flags_reach_child_commands(tmp_path: Path) -> None:
    args = make_args(tmp_path, "--force", "--recreate-serving")
    by_name = {stage.name: stage for stage in build_stages(args, tmp_path)}

    assert "--force" in by_name["openclip_keyframes"].command
    assert "--screening-force" in by_name["ocr"].command
    assert "--recreate" in by_name["elasticsearch"].command
    assert "--drop-existing" in by_name["milvus_keyframes"].command
    assert "--drop-existing" in by_name["milvus_rois"].command
