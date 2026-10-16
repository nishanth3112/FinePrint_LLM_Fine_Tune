import json
from pathlib import Path

DEFAULT_THEME_PATH = Path(__file__).resolve().parents[2] / "configs" / "theme_sic.json"


def load_theme_table(path: str | Path | None = None) -> dict[str, frozenset[str]]:
    """Theme -> SIC codes. An approximation; see the note inside the JSON."""
    raw = json.loads(Path(path or DEFAULT_THEME_PATH).read_text())
    return {
        theme: frozenset(entry["code"] for entry in entries)
        for theme, entries in raw["themes"].items()
    }
