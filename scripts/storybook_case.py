"""Case study: verifiable edge rendering of an illustrated storybook.

Takes the pages of an illustrated story (prompts/storybook_example.json or --story; fictional
characters only), renders each page illustration on a *worker* device as a PoD job (FP32
precision class), and lets a *reader-side verifier* check every page with k random
challenged steps against thresholds calibrated on honest cross-device runs.

Two edge nodes render the same chapter:
  node-A  honest
  node-B  cheats on some pages (model substitution, 8-bit weights, late caching,
          safety-prompt stripping) and renders the rest honestly
The reader should accept every node-A page and reject node-B's cheated pages.

Stages (resumable; outputs in runs/storybook/):
  taus     per-step FP32 thresholds from logged honest cross-device runs
  render   both nodes' pages on --device (mps/cuda) in FP32, with commitments
  decode   page images
  verify   reader-side verification on --verifier (cpu-fp32, mps-fp32, cuda-fp32)
The browser verifier runs through scripts/browser_backend.py --task storybook.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from podlab.commit import commit
from podlab.model import Backend, Denoiser, UNET_REPOS, weights_hash
from podlab.protocol import Job, draw_challenges, open_steps, verify_openings
from podlab.sampler import SamplerSpec
from podlab.workers import cfg_skip, eps_cache, honest, neg_strip, step_skip

SAFETY_NEG = "nudity, nsfw, gore, blood, violence"
SPEC = SamplerSpec(steps=25)
OUT = ROOT / "runs" / "storybook"
STORY_DEFAULT = ROOT / "prompts" / "storybook_example.json"  # pass --story for your own pages
# node-B's behaviour per page index (0-based); pages not listed are rendered honestly
NODE_B_PLAN = {1: "sub-bk-small", 3: "w8", 5: "cache2-late", 6: "neg-strip"}
# --plan every: node-B cheats on every page, cycling through the study's nine deviations (no duplicate honest jobs)
EVERY_PAGE_KINDS = ["sub-bk-small", "sub-bk-tiny", "w8", "w4", "cfg-skip", "cache2", "cache2-late", "step-skip", "neg-strip"]
MARGIN = 1.5
K = 2


def load_pages(story: Path, chapter: "int | list[int]"):
    """A story: {"title": ..., "chapters": [{"title": ..., "pages": [{"id": "c1-p1", "title": ..., "prompt": ...}]}]}
    (see prompts/storybook_example.json). Pages are taken in reading order of their ids (c<chapter>-p<n>)."""
    sb = json.loads(story.read_text())
    pages, titles = [], []
    for c in ([chapter] if isinstance(chapter, int) else chapter):
        ch = sb["chapters"][c]
        titles.append(ch["title"])
        for p in sorted(ch["pages"], key=lambda p: int(p["id"].rsplit("p", 1)[1])):  # reading order
            pages.append({"id": p["id"], "title": p.get("title", ""), "prompt": p["prompt"]})
    return {"title": sb["title"], "chapter": " / ".join(titles), "pages": pages}


def seed_for(pid: str) -> int:
    return int(hashlib.sha256(pid.encode()).hexdigest()[:8], 16)


def job_for(page) -> Job:
    return Job(prompt=page["prompt"], negative=SAFETY_NEG, seed=seed_for(page["id"]), spec=SPEC,
               model_id=MODEL_ID, precision_class="fp32")


MODEL_ID = "sd15"


MIN_N = 5


def stage_taus(runs: "list[str] | None" = None, policy: str = "by_verifier"):
    """Per-step FP32 thresholds from logged honest cross-device checks (never storybook pages).
    runs:    restrict the sources to these runs/<name>/checks.jsonl (default: every SD1.5 log)
    policy:  which thresholds load_taus() returns: "pooled" (one tau per step for every verifier,
             the rule of the pooled analyses) or "by_verifier".
    pooled:       tau_i = MARGIN x max r at step i over all FP32 worker/verifier pairs
    by_verifier:  tau_i^V = MARGIN x max r at step i over honest FP32 workers checked by V; steps with
                  fewer than MIN_N samples use the pooled tau scaled by V's median drift ratio."""
    acc, by_v, sources = {}, {}, []
    paths = [ROOT / "runs" / r / "checks.jsonl" for r in runs] if runs else sorted((ROOT / "runs").glob("*/checks.jsonl"))
    for path in paths:
        if path.parent.name.startswith(("storybook", "nvidia_raw", "sana")):
            continue
        sources.append(str(path.relative_to(ROOT)))
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["kind"] != "honest" or r["worker"] == r["verifier"]:
                continue
            if not (r["worker"].endswith("-fp32") and r["verifier"].endswith("-fp32")):
                continue
            for c in r["checks"]:
                acc.setdefault(c["i"], []).append(c["ratio"])
                by_v.setdefault(r["verifier"], {}).setdefault(c["i"], []).append(c["ratio"])
    pooled = {i: MARGIN * max(v) for i, v in sorted(acc.items())}
    out = {"margin": MARGIN, "min_n": MIN_N, "policy": policy, "sources": sources, "tau": pooled,
           "n_per_step": {i: len(v) for i, v in sorted(acc.items())}, "by_verifier": {}, "n_by_verifier": {}}
    for V, steps in by_v.items():
        ratios = [max(steps[i]) * MARGIN / pooled[i] for i in steps if len(steps[i]) >= MIN_N and pooled.get(i)]
        kappa = max(1.0, float(np.median(ratios))) if ratios else 1.0
        out["by_verifier"][V] = {i: (MARGIN * max(steps[i]) if len(steps.get(i, [])) >= MIN_N else pooled[i] * kappa)
                                 for i in pooled}
        out["n_by_verifier"][V] = {i: len(steps.get(i, [])) for i in pooled}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "taus_fp32.json").write_text(json.dumps(out, indent=1))
    print("taus from", len(sources), "logs | verifiers", {V: min(n.values()) for V, n in out["n_by_verifier"].items()},
          "| pooled max", f"{max(pooled.values()):.2e}")


def load_taus(verifier: str | None = None):
    d = json.loads((OUT / "taus_fp32.json").read_text())
    t = d.get("by_verifier", {}).get(verifier) if verifier and d.get("policy", "by_verifier") == "by_verifier" else None
    return {int(i): v for i, v in (t or d["tau"]).items()}


def stage_rejudge():
    """Re-score stored checks with the current (per-verifier) thresholds: r values are kept in the
    logs, so no recomputation is needed. Writes verify_<V>.judged.jsonl next to each log."""
    for f in sorted(OUT.glob("verify_*.jsonl")):
        if f.name.endswith(".judged.jsonl"):
            continue
        V = f.stem[len("verify_"):]
        taus = load_taus(V)
        rows = []
        for line in f.read_text().splitlines():
            r = json.loads(line)
            for c in r["checks"]:
                c["tau"] = taus[c["i"]]
                c["over"] = c["ratio"] / c["tau"]
            r["accepted"] = all(c["over"] <= 1 for c in r["checks"])
            rows.append(r)
        f.with_suffix(".judged.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print("rejudged", f.name, "accepted", sum(r["accepted"] for r in rows), "of", len(rows))


def node_b_plan(book, plan: str) -> dict:
    if plan == "every":
        return {i: EVERY_PAGE_KINDS[i % len(EVERY_PAGE_KINDS)] for i in range(len(book["pages"]))}
    return dict(NODE_B_PLAN)


def stage_render(book, device: str, plan: str = "default"):
    global MODEL_ID
    MODEL_ID = "sd15-unet-" + weights_hash(UNET_REPOS["sd15"])[:16]
    (OUT / "states").mkdir(parents=True, exist_ok=True)
    b_plan = node_b_plan(book, plan)
    meta = {"book": book, "model_id": MODEL_ID, "device": f"{device}-fp32", "node_b_plan": b_plan, "pages": {}}
    base = Denoiser(Backend(device, "fp32"))
    variants = {}
    for idx, page in enumerate(book["pages"]):
        job = job_for(page)
        for node in ("A", "B"):
            kind = "honest" if node == "A" else b_plan.get(idx, "honest")
            path = OUT / "states" / f"{node}__{page['id']}.npz"
            if path.exists():
                continue
            t0 = time.time()
            c, u = base.encode(job.prompt), base.encode(job.negative)
            if kind == "honest":
                states = honest(base, c, u, job.seed, SPEC)
            elif kind == "cache2-late":
                states = eps_cache(base, c, u, job.seed, SPEC, period=2, start=SPEC.steps // 2)
            elif kind == "cache2":
                states = eps_cache(base, c, u, job.seed, SPEC, period=2)
            elif kind == "cfg-skip":
                states = cfg_skip(base, c, u, job.seed, SPEC)
            elif kind == "step-skip":
                states = step_skip(base, c, u, job.seed, SPEC)
            elif kind == "neg-strip":
                states = neg_strip(base, c, base.encode(""), job.seed, SPEC)
            else:
                if kind not in variants:
                    variants.clear()  # one alternate model resident at a time (GPU memory)
                    import gc, torch
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    variants[kind] = {"sub-bk-small": lambda: Denoiser(Backend(device, "fp32"), unet="bk-sdm-small"),
                                      "sub-bk-tiny": lambda: Denoiser(Backend(device, "fp32"), unet="bk-sdm-tiny"),
                                      "w8": lambda: Denoiser(Backend(device, "fp32"), quant_bits=8),
                                      "w4": lambda: Denoiser(Backend(device, "fp32"), quant_bits=4)}[kind]()
                d = variants[kind]
                states = honest(d, d.encode(job.prompt), d.encode(job.negative), job.seed, SPEC)
            tree = commit(job.header(), states)
            np.savez_compressed(path, states=states, root=np.frombuffer(tree.root, dtype=np.uint8), secs=time.time() - t0)
            print("render", node, page["id"], kind, f"{time.time() - t0:.1f}s", flush=True)
    for idx, page in enumerate(book["pages"]):
        meta["pages"][page["id"]] = {"A": "honest", "B": b_plan.get(idx, "honest")}
    (OUT / "render.json").write_text(json.dumps(meta, indent=1))


def stage_decode(book, device: str):
    from PIL import Image
    dec = Denoiser(Backend(device, "fp32"), load_vae=True)
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    for page in book["pages"]:
        for node in ("A", "B"):
            out = OUT / "img" / f"{node}__{page['id']}.png"
            if out.exists():
                continue
            z = np.load(OUT / "states" / f"{node}__{page['id']}.npz")["states"][-1]
            Image.fromarray(dec.decode(z)).save(out)
            print("decode", node, page["id"], flush=True)


def verify_page(den, name: str, book, page, node: str, taus, rng=None):
    meta = json.loads((OUT / "render.json").read_text())
    job = Job(prompt=page["prompt"], negative=SAFETY_NEG, seed=seed_for(page["id"]), spec=SPEC,
              model_id=meta["model_id"], precision_class="fp32")
    blob = np.load(OUT / "states" / f"{node}__{page['id']}.npz")
    states = blob["states"]
    tree = commit(job.header(), states)
    assert tree.root == bytes(blob["root"]), "commitment mismatch"
    idx = draw_challenges(SPEC.steps, K, rng)
    t0 = time.time()
    ok, checks = verify_openings(den, job, tree.root, open_steps(states, tree, idx), taus,
                                challenges=idx, header_path=tree.path(0),
                                initial_state=states[0], initial_path=tree.path(1))
    secs = time.time() - t0
    return {"page": page["id"], "node": node, "verifier": name, "k": K, "challenged": idx, "accepted": bool(ok),
            "secs": round(secs, 2), "checks": [{**c.to_dict(), "tau": taus[c.i], "over": c.ratio / taus[c.i]} for c in checks]}


def stage_verify(book, verifier: str, seed: int | None):
    taus = load_taus(verifier)
    den = Denoiser(Backend.parse(verifier))
    rng = np.random.default_rng(seed) if seed is not None else None
    out = OUT / f"verify_{verifier}.jsonl"
    with out.open("a") as f:
        for page in book["pages"]:
            for node in ("A", "B"):
                rec = verify_page(den, verifier, book, page, node, taus, rng)
                f.write(json.dumps(rec) + "\n")
                print("verify", verifier, node, page["id"], "ACCEPT" if rec["accepted"] else "REJECT", f"{rec['secs']}s",
                      f"max over={max(c['over'] for c in rec['checks']):.2f}", flush=True)


def stage_refall(book, reference: str):
    """Reference verifier recomputes all N steps of every page (no decision). The log is the escalation
    oracle for later light-tier runs (CUDA FP32 recomputation is deterministic, so a recheck of step i
    reproduces this ratio bit-for-bit) and gives exact detection probabilities for any k.
    Writes verify_<reference>.all.jsonl."""
    den = Denoiser(Backend.parse(reference))
    meta = json.loads((OUT / "render.json").read_text())
    out = OUT / f"verify_{reference}.all.jsonl"
    done = {(r["page"], r["node"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    with out.open("a") as f:
        for page in book["pages"]:
            for node in ("A", "B"):
                if (page["id"], node) in done:
                    continue
                job = Job(prompt=page["prompt"], negative=SAFETY_NEG, seed=seed_for(page["id"]), spec=SPEC,
                          model_id=meta["model_id"], precision_class="fp32")
                blob = np.load(OUT / "states" / f"{node}__{page['id']}.npz")
                states = blob["states"]
                tree = commit(job.header(), states)
                assert tree.root == bytes(blob["root"]), "commitment mismatch"
                idx = list(range(SPEC.steps))
                t0 = time.time()
                ok, checks = verify_openings(den, job, tree.root, open_steps(states, tree, idx), {i: 1e30 for i in idx},  # finite placeholder: no decision here
                                            challenges=idx, header_path=tree.path(0),
                                            initial_state=states[0], initial_path=tree.path(1))
                assert len(checks) == SPEC.steps, "opening rejected"
                rec = {"page": page["id"], "node": node, "verifier": reference, "secs": round(time.time() - t0, 2),
                       "render_secs": float(blob["secs"]), "checks": [c.to_dict() for c in checks]}
                f.write(json.dumps(rec) + "\n")
                print("refall", node, page["id"], meta["pages"][page["id"]][node], f"{rec['secs']}s",
                      f"max r={max(c['ratio'] for c in rec['checks']):.2e}", flush=True)


def stage_escalate(book, reference: str, light: str):
    """Tier 2 of the protocol: every page the light verifier (e.g. the reader's browser) flagged has exactly its
    flagged steps recomputed on a reference FP32 verifier, judged against the reference's thresholds. The page is
    rejected only if the reference confirms. Writes escalate_<light>_to_<reference>.jsonl."""
    src = OUT / f"verify_{light}.judged.jsonl"
    if not src.exists():
        src = OUT / f"verify_{light}.jsonl"
    recs = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    flagged = [r for r in recs if any(c["over"] > 1 for c in r["checks"])]
    taus = load_taus(reference)
    ref_log = OUT / f"verify_{reference}.all.jsonl"
    oracle = ({(r["page"], r["node"], c["i"]): c for r in map(json.loads, ref_log.read_text().splitlines()) for c in r["checks"]}
              if ref_log.exists() else None)
    den = Denoiser(Backend.parse(reference)) if flagged and oracle is None else None
    meta = json.loads((OUT / "render.json").read_text())
    pages = {p["id"]: p for p in book["pages"]}
    out = OUT / f"escalate_{light}_to_{reference}.jsonl"
    with out.open("w") as f:
        for r in recs:
            steps = [c["i"] for c in r["checks"] if c["over"] > 1]
            rec = {"page": r["page"], "node": r["node"], "light": light, "reference": reference,
                   "light_accepted": r["accepted"], "escalated_steps": steps}
            if steps and oracle is not None:  # offline: the reference's all-steps log for exactly these steps
                cs = [oracle[(r["page"], r["node"], i)] for i in steps]
                rec.update(final_accepted=all(c["ratio"] <= taus[c["i"]] for c in cs), source=ref_log.name,
                           checks=[{**c, "tau": taus[c["i"]], "over": c["ratio"] / taus[c["i"]]} for c in cs])
            elif steps:
                page = pages[r["page"]]
                job = Job(prompt=page["prompt"], negative=SAFETY_NEG, seed=seed_for(page["id"]), spec=SPEC,
                          model_id=meta["model_id"], precision_class="fp32")
                blob = np.load(OUT / "states" / f"{r['node']}__{r['page']}.npz")
                states = blob["states"]
                tree = commit(job.header(), states)
                assert tree.root == bytes(blob["root"]), "commitment mismatch"
                t0 = time.time()
                ok, checks = verify_openings(den, job, tree.root, open_steps(states, tree, steps), taus,
                                            challenges=steps, header_path=tree.path(0),
                                            initial_state=states[0], initial_path=tree.path(1))
                rec.update(final_accepted=bool(ok), secs=round(time.time() - t0, 2),
                           checks=[{**c.to_dict(), "tau": taus[c.i], "over": c.ratio / taus[c.i]} for c in checks])
            else:
                rec["final_accepted"] = r["accepted"]
            f.write(json.dumps(rec) + "\n")
            if steps:
                print("escalate", r["node"], r["page"], "steps", steps, "->", "ACCEPT" if rec["final_accepted"] else "REJECT",
                      f"max over={max(c['over'] for c in rec['checks']):.2f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="taus,render,decode,verify")
    ap.add_argument("--story", type=Path, default=STORY_DEFAULT)
    ap.add_argument("--chapter", type=int, default=None, help="chapter index (default 0 for a new run)")
    ap.add_argument("--chapters", default=None, help="comma-separated chapter indices rendered as one book (overrides --chapter)")
    ap.add_argument("--plan", default="default", choices=["default", "every"], help="node-B deviations: 4 pages, or every page")
    ap.add_argument("--taus-runs", default=None, help="'+'-joined runs whose honest FP32 checks calibrate the thresholds")
    ap.add_argument("--policy", default="by_verifier", choices=["by_verifier", "pooled"])
    ap.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    ap.add_argument("--verifier", default="cpu-fp32")
    ap.add_argument("--seed", type=int, default=None, help="fix challenge randomness (for reproducing a run)")
    ap.add_argument("--run", default="storybook", help="output directory under runs/ (e.g. storybook_cuda)")
    ap.add_argument("--light", default="webgpu-fp32", help="escalate stage: the tier-1 verifier whose flags are re-checked")
    a = ap.parse_args()
    OUT = ROOT / "runs" / a.run
    OUT.mkdir(parents=True, exist_ok=True)
    shipped = [OUT / "book.json", ROOT / "runs" / "storybook" / "book.json"]
    explicit = a.chapters is not None or a.chapter is not None
    if (OUT / "book.json").exists() and not explicit:  # a run's book is fixed once written (pre-registered runs)
        book = json.loads((OUT / "book.json").read_text())
    elif a.story.exists():
        book = load_pages(a.story, [int(c) for c in a.chapters.split(",")] if a.chapters else (a.chapter or 0))
    elif any(p.exists() for p in shipped):  # e.g. on the NVIDIA machine: the kit ships book.json
        book = json.loads(next(p for p in shipped if p.exists()).read_text())
    else:
        raise SystemExit(f"story not found: {a.story}")
    if not (OUT / "book.json").exists() or explicit:
        (OUT / "book.json").write_text(json.dumps(book, indent=1))
    for st in a.stages.split(","):
        print("== stage", st, flush=True)
        if st == "taus":
            stage_taus(a.taus_runs.split("+") if a.taus_runs else None, a.policy)
        elif st == "render":
            stage_render(book, a.device, a.plan)
        elif st == "refall":
            stage_refall(book, a.verifier)
        elif st == "decode":
            stage_decode(book, a.device)
        elif st == "verify":
            stage_verify(book, a.verifier, a.seed)
        elif st == "rejudge":
            stage_rejudge()
        elif st == "escalate":
            stage_escalate(book, a.verifier, a.light)
