from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def add_project_root_to_path() -> None:
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_path(path: str | Path, base: Path = PROJECT_ROOT) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return base / candidate
