"""Sample a calibration prompt set from PartiPrompts that is disjoint from the study's 48 prompts
(prompts/parti_sample.json) and from every test page. Same filters as make_prompts.py (no People or
World Knowledge, at most 40 words), stratified round-robin over categories, fixed seed.

    python scripts/make_calib_prompts.py [N]          # default 240
Writes prompts/parti_calib.json. Used only to calibrate thresholds (request-level split conformal).
"""
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {"People", "World Knowledge"}
N, SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 240, 2028

taken = {p["prompt"] for p in json.loads((ROOT / "prompts" / "parti_sample.json").read_text())}
rows = list(csv.DictReader(open(ROOT / "prompts" / "PartiPrompts.tsv"), delimiter="\t"))
by, seen = defaultdict(list), set()
for r in rows:
    p = r["Prompt"].strip()
    if r["Category"] in EXCLUDE or len(p.split()) > 40 or p in taken or p in seen:
        continue
    seen.add(p)
    by[r["Category"]].append({"prompt": p, "category": r["Category"], "challenge": r["Challenge"]})
rng = random.Random(SEED)
for c in by:
    rng.shuffle(by[c])
cats, out, i = sorted(by), [], 0
while len(out) < N and any(by.values()):
    c = cats[i % len(cats)]
    if by[c]:
        out.append(by[c].pop())
    i += 1
assert not taken & {o["prompt"] for o in out}
(ROOT / "prompts" / "parti_calib.json").write_text(json.dumps(out, indent=1))
print(len(out), "calibration prompts from", len(cats), "categories (disjoint from parti_sample.json)")
