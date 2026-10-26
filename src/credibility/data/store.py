"""Versioned, hash-verified price snapshots.

Layout::

    data/snapshots/<name>/
        manifest.json          config, per-symbol hash/rows/dates, quality reports
        prices/<SYMBOL>.parquet

The hash is computed over a canonical CSV rendering of the data (not the parquet
bytes), so it does not depend on the pyarrow version.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from credibility import config as _config

MANIFEST = "manifest.json"


class SnapshotError(RuntimeError):
    pass


def content_hash(df: pd.DataFrame) -> str:
    canonical = df.to_csv(index=False, date_format="%Y-%m-%d", float_format="%.10g")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _safe(symbol: str) -> str:
    return symbol.replace("^", "_IDX_").replace("/", "_")


@dataclass
class Snapshot:
    root: Path
    manifest: dict

    @classmethod
    def path_for(cls, name: str, base: Path | None = None) -> Path:
        return Path(base if base is not None else _config.SNAPSHOT_DIR) / name

    @classmethod
    def create(cls, name: str, config: dict, base: Path | None = None, overwrite: bool = False) -> "Snapshot":
        root = cls.path_for(name, base)
        if (root / MANIFEST).exists() and not overwrite:
            raise SnapshotError(f"snapshot '{name}' already exists; snapshots are immutable (use a new name)")
        (root / "prices").mkdir(parents=True, exist_ok=True)
        manifest = {
            "name": name,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": config,
            "symbols": {},
            "failed": {},
        }
        snap = cls(root, manifest)
        snap.save_manifest()
        return snap

    @classmethod
    def open(cls, name: str, base: Path | None = None) -> "Snapshot":
        root = cls.path_for(name, base)
        mpath = root / MANIFEST
        if not mpath.exists():
            raise SnapshotError(f"no snapshot named '{name}' at {root}")
        return cls(root, json.loads(mpath.read_text(encoding="utf-8")))

    def save_manifest(self) -> None:
        (self.root / MANIFEST).write_text(json.dumps(self.manifest, indent=2, sort_keys=True), encoding="utf-8")

    def _file(self, symbol: str) -> Path:
        return self.root / "prices" / f"{_safe(symbol)}.parquet"

    def add(self, symbol: str, df: pd.DataFrame, quality: dict) -> None:
        df.to_parquet(self._file(symbol), index=False)
        self.manifest["symbols"][symbol] = {
            "file": self._file(symbol).name,
            "sha256": content_hash(df),
            "rows": int(len(df)),
            "start": df["date"].min().strftime("%Y-%m-%d"),
            "end": df["date"].max().strftime("%Y-%m-%d"),
            "quality": quality,
        }
        self.save_manifest()

    def add_failure(self, symbol: str, reason: str) -> None:
        self.manifest["failed"][symbol] = reason
        self.save_manifest()

    def symbols(self, only_passed: bool = True) -> list[str]:
        out = []
        for sym, meta in self.manifest["symbols"].items():
            if only_passed and not meta["quality"]["passed"]:
                continue
            out.append(sym)
        return out

    def load(self, symbol: str, verify: bool = True) -> pd.DataFrame:
        meta = self.manifest["symbols"].get(symbol)
        if meta is None:
            raise SnapshotError(f"{symbol} not in snapshot '{self.manifest['name']}'")
        df = pd.read_parquet(self._file(symbol))
        if verify and content_hash(df) != meta["sha256"]:
            raise SnapshotError(f"hash mismatch for {symbol}: snapshot was modified")
        return df

    def verify_all(self) -> dict[str, bool]:
        return {s: content_hash(pd.read_parquet(self._file(s))) == m["sha256"] for s, m in self.manifest["symbols"].items()}
