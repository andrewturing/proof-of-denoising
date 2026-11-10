"""Sample a stratified, interleaved prompt set from PartiPrompts (Apache-2.0,
github.com/google-research/parti). Excludes People and World Knowledge so the study
never renders real persons. Writes prompts/parti_sample.json."""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {"People", "World Knowledge"}
PER_CAT, TOTAL, SEED = 5, 48, 2027

rows = list(csv.DictReader(open(ROOT / "prompts" / "PartiPrompts.tsv"), delimiter="\t"))
by = defaultdict(list)
for r in rows:
    if r["Category"] not in EXCLUDE and len(r["Prompt"].split()) <= 40:
        by[r["Category"]].append(r)
rng = random.Random(SEED)
picks = {c: rng.sample(v, PER_CAT) for c, v in sorted(by.items())}
cats = sorted(picks)
out = [{"prompt": picks[c][i]["Prompt"], "category": c, "challenge": picks[c][i]["Challenge"]}
       for i in range(PER_CAT) for c in cats][:TOTAL]
(ROOT / "prompts" / "parti_sample.json").write_text(json.dumps(out, indent=1))
print(len(out), "prompts from", len(cats), "categories:", cats)
for o in out[:10]:
    print("-", o["category"], "|", o["prompt"])
