"""Fingerprint the ORIGINAL study files and verify later that none of them changed.

    python scripts/protect_check.py --init      record SHA-256 of every protected file (refuses to overwrite)
    python scripts/protect_check.py --verify    re-hash and compare; exit 1 if anything changed or disappeared

Protected: the frozen data snapshot, the four stored result runs (+ holdout locks), the registry and its evidence,
the configs, and the paper's already-generated tables/figures/numbers. New files are allowed.
"""
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
STORE = ROOT / "protected_hashes.json"
PATTERNS = [
    "data/snapshots/nse_v1/**/*",
    "results/ret_dev/*", "results/vol_dev/*", "results/ret_hold/*", "results/vol_hold/*",
    "results/holdout_return.lock", "results/holdout_volatility.lock",
    "registry/**/*",
    "configs/data.yaml", "configs/eval.yaml",
    "paper/generated/*",
]


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def files():
    seen = {}
    for pat in PATTERNS:
        for f in ROOT.glob(pat):
            if f.is_file():
                seen[f.relative_to(ROOT).as_posix()] = f
    return seen


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--verify"
    if mode == "--init":
        if STORE.exists():
            print("baseline already exists; refusing to overwrite")
            return 1
        h = {k: sha(f) for k, f in sorted(files().items())}
        STORE.write_text(json.dumps(h, indent=1), encoding="utf-8")
        print(f"recorded {len(h)} protected files")
        return 0
    base = json.loads(STORE.read_text(encoding="utf-8"))
    cur = files()
    changed = [k for k, v in base.items() if k in cur and sha(cur[k]) != v]
    missing = [k for k in base if k not in cur]
    print(f"protected files: {len(base)} | changed: {len(changed)} | missing: {len(missing)}")
    for k in changed + missing:
        print("   !!", k)
    return 1 if (changed or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
