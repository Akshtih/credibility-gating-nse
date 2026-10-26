"""Build the two deliverables and scan them for personal paths / key-like text.

    python scripts/make_package.py

stock-credibility_reproducibility.zip   code, configs, frozen data, stored forecasts, registry, scripts, tests, docs for reviewers
paper_overleaf.zip                      main.tex + IEEEtran.cls + generated tables/figures (upload to Overleaf)
"""
import pathlib
import re
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", "__pycache__", ".pytest_cache", ".git", "paper", "docs", ".github", "stock-credibility_reproducibility"}
SKIP_FILES = {"paper_overleaf.zip", "stock-credibility_reproducibility.zip", "Dockerfile", ".dockerignore"}
TEXT = (".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".json", ".csv", ".html", ".tex")
RISKY = re.compile(r"(C:\\\\Users|C:/Users|AppData|gsk_[A-Za-z0-9]{10,}|api[_-]?key\s*=\s*['\"][^'\"]+|password\s*=)", re.I)


def build() -> int:
    repro, paper = ROOT / "stock-credibility_reproducibility.zip", ROOT / "paper_overleaf.zip"
    for z in (repro, paper):
        z.unlink(missing_ok=True)
    with zipfile.ZipFile(repro, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(ROOT.rglob("*")):
            rel = f.relative_to(ROOT)
            if (f.is_dir() or any(p in SKIP_DIRS for p in rel.parts) or f.name in SKIP_FILES
                    or f.suffix in {".pyc", ".log", ".flag"} or "egg-info" in str(rel) or f.name == ".env"):
                continue
            z.write(f, "stock-credibility/" + rel.as_posix())
        for f in sorted((ROOT / "paper" / "generated").iterdir()):
            z.write(f, "stock-credibility/paper/generated/" + f.name)
    p = ROOT / "paper"
    with zipfile.ZipFile(paper, "w", zipfile.ZIP_DEFLATED) as z:
        for f in [p / "main.tex", p / "IEEEtran.cls", *sorted((p / "generated").iterdir())]:
            z.write(f, f.relative_to(p).as_posix())
    bad = []
    for zp in (repro, paper):
        zf = zipfile.ZipFile(zp)
        for n in zf.namelist():
            if n.endswith(TEXT) and not n.endswith("make_package.py") and RISKY.search(zf.read(n).decode("utf-8", "ignore")):
                bad.append((zp.name, n))
    print(f"{repro.name}: {repro.stat().st_size / 1e6:.1f} MB | {paper.name}: {paper.stat().st_size / 1e3:.0f} KB")
    print("personal-path / key-like text found in:" if bad else "scan clean: no personal paths or key-like text", *bad, sep="\n  ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(build())
