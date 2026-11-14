"""Process configuration for the local, deidentified prototype."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database: Path
    ai_root: Path | None = None
    token: str = ""
    actor: str = "local-user"

    @classmethod
    def from_env(cls) -> "Settings":
        root = os.environ.get("SEWS_AI_REPO_ROOT", "").strip()
        return cls(
            database=Path(os.environ.get("SEWS_DATABASE_PATH", "data/sews.sqlite"))
            .expanduser()
            .resolve(),
            ai_root=Path(root).expanduser().resolve() if root else None,
            token=os.environ.get("SEWS_API_TOKEN", ""),
            actor=os.environ.get("SEWS_ACTOR", "local-user").strip() or "local-user",
        )
