"""File-based model registry with an audit trail.

    registry/
        index.json                 {"<task>": {"<horizon>": "<entry id>"}}   (current served model per task/horizon)
        entries/<id>.json          immutable entry (content-addressed)

An entry records what is needed to reproduce and justify the decision: model name + parameters, the data
snapshot and a hash of its manifest, the config hash, the dev run it was evaluated on, the gate metrics,
the git commit, and the role (champion / fallback / rejected). Serving re-fits the registered spec on the
latest snapshot data (fits take well under a second), so no pickled weights go stale.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from credibility.config import PROJECT_ROOT


def git_commit(root: Path = PROJECT_ROOT) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unversioned"
    except Exception:
        return "unversioned"


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Registry:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        (self.root / "entries").mkdir(parents=True, exist_ok=True)

    @property
    def _index_path(self) -> Path:
        return self.root / "index.json"

    def index(self) -> dict:
        if not self._index_path.exists():
            return {}
        return json.loads(self._index_path.read_text(encoding="utf-8"))

    def register(self, entry: dict, serve: bool = False) -> str:
        """Write an immutable entry; if ``serve``, make it the served model for its (task, horizon)."""
        body = dict(entry)
        body.setdefault("created_utc", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        body.setdefault("git_commit", git_commit())
        canon = json.dumps({k: v for k, v in body.items() if k != "created_utc"}, sort_keys=True, default=str)
        entry_id = hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]
        body["id"] = entry_id
        (self.root / "entries" / f"{entry_id}.json").write_text(json.dumps(body, indent=2, sort_keys=True, default=str),
                                                               encoding="utf-8")
        if serve:
            idx = self.index()
            idx.setdefault(body["task"], {})[str(body["horizon"])] = entry_id
            self._index_path.write_text(json.dumps(idx, indent=2, sort_keys=True), encoding="utf-8")
        return entry_id

    def get(self, entry_id: str) -> dict:
        return json.loads((self.root / "entries" / f"{entry_id}.json").read_text(encoding="utf-8"))

    def served(self, task: str, horizon: int) -> dict | None:
        eid = self.index().get(task, {}).get(str(horizon))
        return self.get(eid) if eid else None

    def entries(self) -> list[dict]:
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((self.root / "entries").glob("*.json"))]
