"""Local self-overlap check: shared word n-grams between the new paper and the team's earlier materials.

This is NOT a plagiarism checker (it cannot see the web or journals). It only measures overlap with
text we have on disk: the old report (.docx), the slide decks (.pptx) and the old paper drafts (.tex).
"""
import pathlib
import re
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT.parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def tex_to_text(path: pathlib.Path) -> str:
    t = path.read_text(encoding="utf-8", errors="ignore")
    t = re.sub(r"(?m)^\s*%.*$", " ", t)
    t = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", " ", t, flags=re.S)
    t = re.sub(r"\\(?:cite|ref|label|input|includegraphics)(\[[^\]]*\])?\{[^}]*\}", " ", t)
    t = re.sub(r"\\num[A-Za-z]+\{?\}?", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    return re.sub(r"[{}$~\\&%#_^]", " ", t)


def docx_text(path):
    x = zipfile.ZipFile(path).read("word/document.xml").decode("utf8", "ignore")
    return " ".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", x))


def pptx_text(path):
    z = zipfile.ZipFile(path)
    return " ".join(" ".join(re.findall(r"<a:t>([^<]*)</a:t>", z.read(n).decode("utf8", "ignore")))
                    for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml", n))


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def grams(w, n):
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


new_w = words(tex_to_text(ROOT / "paper" / "main.tex"))
new_g = grams(new_w, N)
sources = {}
for p in (BASE / "Materials").glob("*.docx"):
    sources[p.name] = words(docx_text(p))
for p in (BASE / "Materials").glob("*.pptx"):
    sources[p.name] = words(pptx_text(p))
for p in [BASE / "paper" / "main.tex", BASE / "IEEE_Conference_Template__1_" / "conference_101719.tex"]:
    if p.exists():
        sources[str(p.relative_to(BASE))] = words(tex_to_text(p))
readme = BASE / "Stock-Market-Prediction-MLOps" / "README.md"
if readme.exists():
    sources["old README"] = words(readme.read_text(encoding="utf-8", errors="ignore"))

print(f"new paper: {len(new_w)} words, {len(new_g)} distinct {N}-grams")
total_hit = set()
for name, w in sorted(sources.items()):
    hit = new_g & grams(w, N)
    total_hit |= hit
    print(f"  {name:45s} shared {N}-grams: {len(hit):4d}  ({100 * len(hit) / max(len(new_g), 1):.2f}% of paper)")
print(f"UNION shared with any earlier material: {len(total_hit)} ({100 * len(total_hit) / max(len(new_g), 1):.2f}%)")
for g in sorted(total_hit)[:25]:
    print("   -", g)
