"""Static sanity checks for paper/main.tex (use when no LaTeX compiler is available)."""
import pathlib
import re
import sys

root = pathlib.Path(__file__).resolve().parents[1] / "paper"
t = (root / "main.tex").read_text(encoding="utf-8")
n = "".join(p.read_text(encoding="utf-8") for p in (root / "generated").glob("numbers*.tex"))
BS = chr(92)

defined = set(re.findall(re.escape(BS) + r"newcommand\{" + re.escape(BS) + r"(num\w+)\}", n))
used = set(re.findall(re.escape(BS) + r"(num[A-Za-z]+)", t))
missing_macros = sorted(used - defined)
unused_macros = sorted(m for m in defined - used if m.startswith("numSkill") is False and m.startswith("numP") is False)

inputs = re.findall(r"(?:includegraphics(?:\[[^\]]*\])?|input)\{([^}]+)\}", t)
missing_files = [f for f in inputs if not (root / f).exists() and not (root / (f + ".tex")).exists()]

cites = {c.strip() for g in re.findall(re.escape(BS) + r"cite\{([^}]+)\}", t) for c in g.split(",")}
bibs = set(re.findall(re.escape(BS) + r"bibitem\{([^}]+)\}", t))
labels = set(re.findall(re.escape(BS) + r"label\{([^}]+)\}", t))
refs = set(re.findall(re.escape(BS) + r"ref\{([^}]+)\}", t))
begins = re.findall(re.escape(BS) + r"begin\{(\w+\*?)\}", t)
ends = re.findall(re.escape(BS) + r"end\{(\w+\*?)\}", t)

problems = []
if missing_macros: problems.append(f"undefined number macros: {missing_macros}")
if missing_files: problems.append(f"missing files: {missing_files}")
if cites - bibs: problems.append(f"cited but no bibitem: {sorted(cites - bibs)}")
if bibs - cites: problems.append(f"bibitem never cited: {sorted(bibs - cites)}")
if refs - labels: problems.append(f"refs without label: {sorted(refs - labels)}")
if sorted(begins) != sorted(ends): problems.append("begin/end mismatch")
if t.count("{") != t.count("}"): problems.append(f"brace imbalance {t.count('{') - t.count('}')}")
if "REPOSITORY URL TO BE ADDED" in open(root / "main.tex", encoding="utf-8").read():
    print("WARNING: repository URL placeholder still in main.tex (replace the repourl macro before submitting)")
print("macros used:", len(used), "| citations:", len(cites), "| labels:", len(labels), "| figures/tables inputs:", len(inputs))
print("PROBLEMS:" if problems else "no problems found", *problems, sep="\n  ")
sys.exit(1 if problems else 0)
