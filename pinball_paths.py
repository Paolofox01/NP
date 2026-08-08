from __future__ import annotations

from pathlib import Path


def resolve_pinball_asset(script_dir: Path, filename: str) -> Path:
    candidates = [script_dir / filename, script_dir / "Pinball" / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    candidate_list = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find {filename}. Looked in: {candidate_list}")