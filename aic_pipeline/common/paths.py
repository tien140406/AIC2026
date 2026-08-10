from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelinePaths:
    root: Path
    archive: Path
    workspace: Path
    manifests: Path
    reports: Path
    features: Path
    samples: Path
    shots: Path
    keyframes: Path
    transcripts: Path
    audio: Path
    indexes: Path

    @classmethod
    def from_root(cls, root: str | Path) -> PipelinePaths:
        root_path = Path(root).expanduser().resolve()
        workspace = root_path / "workspace"
        return cls(
            root=root_path,
            archive=root_path / "archive",
            workspace=workspace,
            manifests=workspace / "manifests",
            reports=workspace / "reports",
            features=workspace / "features",
            samples=workspace / "samples",
            shots=workspace / "shots",
            keyframes=workspace / "keyframes",
            transcripts=workspace / "transcripts",
            audio=workspace / "audio" / "asr",
            indexes=workspace / "indexes",
        )

    def ensure_output_dirs(self) -> None:
        self.manifests.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)
        self.features.mkdir(parents=True, exist_ok=True)
        self.samples.mkdir(parents=True, exist_ok=True)
        self.shots.mkdir(parents=True, exist_ok=True)
        self.keyframes.mkdir(parents=True, exist_ok=True)
        self.transcripts.mkdir(parents=True, exist_ok=True)
        self.audio.mkdir(parents=True, exist_ok=True)
        self.indexes.mkdir(parents=True, exist_ok=True)

    def transcript_dir(self, video_id: str) -> Path:
        return self.transcripts / video_id

    def asr_audio_path(self, video_id: str) -> Path:
        return self.audio / f"{video_id}.wav"
