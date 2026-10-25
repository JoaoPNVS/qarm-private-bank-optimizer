"""Team information for the About Us tab (data/team.json; photos in assets/team/).

Never raises: a missing or malformed file gives an empty team and a message for the app, and a
missing or unreadable photo falls back to a round avatar with the member's initials.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEAM_FILE = ROOT / "data" / "team.json"
PHOTO_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


@dataclass
class Member:
    name: str
    role: str = ""
    description: str = ""
    photo: str = ""
    linkedin: str = ""


def load_team(path: Path = TEAM_FILE) -> tuple[list[Member], str, str | None]:
    """(members, repository URL, error message or None)."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        members = [Member(**{k: str(v) for k, v in m.items() if k in Member.__dataclass_fields__})
                   for m in raw.get("members", []) if m.get("name")]
        return members, str(raw.get("repository", "")), None
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        return [], "", f"Team information unavailable ({Path(path).name}: {exc})."


def initials(name: str) -> str:
    """Initials of the name, ignoring placeholders such as "[NOM]"."""
    words = [w for w in name.split() if w[:1].isalpha()]
    return "".join(w[0].upper() for w in words[:2]) or "?"


def photo_data_uri(member: Member, root: Path = ROOT) -> str | None:
    """The photo as a data URI (for a round crop in HTML), or None if missing or unreadable."""
    if not member.photo:
        return None
    path = (root / member.photo).resolve()
    kind = PHOTO_TYPES.get(path.suffix.lower())
    try:
        if kind is None or not path.is_file() or root.resolve() not in path.parents:
            return None
        return f"data:{kind};base64,{base64.b64encode(path.read_bytes()).decode()}"
    except OSError:
        return None
