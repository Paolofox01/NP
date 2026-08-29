from __future__ import annotations

from pathlib import Path


def resolve_pinball_asset(script_dir: Path, filename: str) -> Path:
    script_dir = Path(script_dir).resolve()
    seen = set()
    candidates = []

    def add_candidate(path: Path) -> None:
        path = path.resolve(strict=False)
        if path not in seen:
            seen.add(path)
            candidates.append(path)

    for base in [script_dir, script_dir.parent, Path.cwd(), Path.cwd().parent]:
        add_candidate(base / filename)
        add_candidate(base / "Pinball" / filename)
        add_candidate(base / "assets" / filename)

    for parent in [script_dir, *script_dir.parents]:
        add_candidate(parent / filename)
        add_candidate(parent / "Pinball" / filename)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    candidate_list = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find {filename}. Looked in: {candidate_list}")