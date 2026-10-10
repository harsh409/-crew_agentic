"""
Saved sessions - so a line of work survives across runs and can be reopened.

Each session is one JSON file under ~/.crew_agent_sessions/. A session is an
ordered list of turns (task, intent, answer/summary, files changed). This is
what powers --list and --continue, and what interactive mode appends to.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

STORE_DIR = Path.home() / ".crew_agent_sessions"


@dataclass
class Turn:
    task: str
    intent: str
    output: str = ""
    changed_files: List[str] = field(default_factory=list)
    timestamp: str = ""


@dataclass
class Session:
    id: str
    title: str
    project_dir: str
    created_at: str
    updated_at: str
    turns: List[Turn] = field(default_factory=list)

    def add_turn(self, turn: Turn) -> None:
        turn.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.turns.append(turn)
        self.updated_at = turn.timestamp

    def history_brief(self) -> str:
        """The FULL, un-chopped recap of every earlier turn. Nothing is truncated
        here - the token_saver decides what to keep verbatim and what to summarise
        (see token_saver.build_history)."""
        if not self.turns:
            return ""
        return "EARLIER IN THIS SESSION:\n" + format_turns(self.turns)


def format_turns(turns) -> str:
    """Lay out a list of turns in full - task, files changed, and the complete
    result - with no truncation."""
    lines = []
    for i, t in enumerate(turns, 1):
        lines.append(f"Turn {i} ({t.intent}): {t.task}")
        if t.changed_files:
            lines.append(f"   files changed: {', '.join(t.changed_files)}")
        if t.output:
            lines.append(f"   result: {t.output.strip()}")
    return "\n".join(lines)


class SessionStore:
    def __init__(self, directory: Path = STORE_DIR):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, sid: str) -> Path:
        return self.dir / f"{sid}.json"

    def create(self, task: str, project_dir: str) -> Session:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        # Millisecond stamp keeps ids sortable even within the same second.
        sid = time.strftime("%Y%m%d-%H%M%S") + f"{int(time.time() * 1000) % 1000:03d}"
        title = (task[:60] + "…") if len(task) > 60 else task
        return Session(id=sid, title=title, project_dir=str(project_dir),
                       created_at=now, updated_at=now, turns=[])

    def save(self, session: Session) -> None:
        data = asdict(session)
        self._path(session.id).write_text(json.dumps(data, indent=2))

    def list(self) -> List[Session]:
        out = []
        for f in self.dir.glob("*.json"):
            try:
                out.append(self._load_file(f))
            except Exception:
                continue
        out.sort(key=lambda s: s.updated_at, reverse=True)
        return out

    def _load_file(self, path: Path) -> Session:
        data = json.loads(path.read_text())
        data["turns"] = [Turn(**t) for t in data.get("turns", [])]
        return Session(**data)

    def load(self, ref: str) -> Optional[Session]:
        """Load by full id, a unique id prefix, or the 1-based index in --list."""
        sessions = self.list()
        if ref.isdigit():
            idx = int(ref) - 1
            return sessions[idx] if 0 <= idx < len(sessions) else None
        for s in sessions:
            if s.id == ref or s.id.startswith(ref):
                return s
        return None

    def delete(self, sid: str) -> bool:
        p = self._path(sid)
        if p.exists():
            p.unlink()
            return True
        return False
