"""In-browser WebGPU backend for Proof-of-Denoising.

Serves web/index.html (ONNX Runtime Web on WebGPU) on 127.0.0.1 and hands it denoising
jobs. A BrowserDenoiser implements the same interface as podlab.model.Denoiser, so the
browser can act as a worker or a verifier ("webgpu-fp16"). Bound to localhost only.

Tasks:
  smoke  - one guided step; compare with PyTorch CPU FP32 and print the relative error
  pilot  - honest trajectories as a worker, then verify every pilot trajectory on CPU_STEPS
"""
import argparse
import base64
import json
import queue
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from podlab.model import BASE_REPO, local_dir
from podlab.sampler import SamplerSpec, initial_noise, step

WEB = ROOT / "web"
VENDOR = WEB / "node_modules" / "onnxruntime-web" / "dist"
MODELS = ROOT / "runs" / "onnx"

jobs: "queue.Queue[dict]" = queue.Queue()
results: dict = {}
events: dict = {}
ready = threading.Event()
counter = iter(range(1, 10**9))
timings: list = []
page_info: dict = {}
STORY = ROOT / "runs" / "storybook"
story_current: dict = {}
ROOT_REDIRECT: dict = {}


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def translate_path(self, path):
        path = path.split("?")[0]
        if path.startswith("/vendor/"):
            return str(VENDOR / path[len("/vendor/"):])
        if path.startswith("/models/"):
            return str(MODELS / path[len("/models/"):])
        if path in ("/", "/index.html"):
            return str(WEB / "index.html")
        if path in ("/storybook", "/storybook.html"):
            return str(WEB / "storybook.html")
        if path == "/worker.js":
            return str(WEB / "worker.js")
        if path.startswith("/storybook/img/"):
            return str(STORY / "img" / Path(path).name)
        return str(WEB / "__missing__")

    def guess_type(self, path):
        if str(path).endswith((".mjs", ".js")):
            return "text/javascript"
        if str(path).endswith(".wasm"):
            return "application/wasm"
        return super().guess_type(path)

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if ROOT_REDIRECT.get("to") and self.path.split("?")[0] in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", ROOT_REDIRECT["to"])
            self.end_headers()
            return
        if self.path.startswith("/storybook/book"):
            return self._json({"book": json.loads((STORY / "book.json").read_text()),
                               "render": json.loads((STORY / "render.json").read_text())})
        if self.path.startswith("/storybook/results"):
            f = STORY / "verify_webgpu-fp32.jsonl"
            rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
            return self._json(rows)
        if self.path.startswith("/storybook/status"):
            return self._json({"current": story_current.get("job")})
        if self.path.startswith("/next"):
            try:
                job = jobs.get(timeout=20)
            except queue.Empty:
                self.send_response(204)
                self.end_headers()
                return
            body = json.dumps(job).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        if self.path.startswith("/log"):
            msg = body.decode(errors="replace")
            print("[browser]", msg, flush=True)
            if msg.startswith("unet file "):
                page_info["unet"] = msg[len("unet file "):]
        elif self.path.startswith("/ready"):
            ready.set()
        elif self.path.startswith("/result/"):
            jid = int(self.path.split("/")[2].split("?")[0])
            if "error=1" in self.path:
                results[jid] = RuntimeError(body.decode(errors="replace"))
            else:
                results[jid] = np.frombuffer(body, dtype=np.float32).copy()
                if "ms=" in self.path:
                    timings.append(int(self.path.split("ms=")[1]))
            events[jid].set()
        self.send_response(204)
        self.end_headers()


def submit(job: dict) -> np.ndarray:
    jid = next(counter)
    job["id"] = jid
    events[jid] = threading.Event()
    jobs.put(job)
    events[jid].wait()
    out = results.pop(jid)
    if isinstance(out, Exception):
        raise out
    return out


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()


class BrowserDenoiser:
    """Same interface as podlab.model.Denoiser, executed by the WebGPU page."""

    def __init__(self):
        self.name = "webgpu-fp32" if "fp32" in page_info.get("unet", "") else "webgpu-fp16"
        from transformers import CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(local_dir(BASE_REPO) / "tokenizer")

    def encode(self, prompt: str) -> np.ndarray:
        ids = self.tokenizer(prompt, padding="max_length", max_length=77, truncation=True).input_ids
        return submit({"op": "text", "ids": ids}).reshape(77, 768)

    def unet_eps(self, z: np.ndarray, t: int, embs: np.ndarray) -> np.ndarray:
        b = embs.shape[0]
        e = embs if b == 2 else np.repeat(embs[:1], 2, axis=0)
        out = submit({"op": "unet", "t": int(t), "sample": b64(np.repeat(z[None], 2, axis=0)), "embs": b64(e)})
        return out.reshape(2, 4, 64, 64)[:b]

    def eps_fn(self, cond, uncond, guidance):
        embs = np.stack([uncond, cond])

        def f(z, t):
            e_u, e_c = self.unet_eps(z, t, embs)
            return e_u + guidance * (e_c - e_u)

        return f


def task_smoke(den: BrowserDenoiser):
    from podlab.model import Backend, Denoiser
    spec = SamplerSpec()
    z = initial_noise(0, spec)
    cpu = Denoiser(Backend("cpu", "fp32"))
    prompt = "a red bicycle leaning against a brick wall"
    c_w, u_w = den.encode(prompt), den.encode("")
    c_c, u_c = cpu.encode(prompt), cpu.encode("")
    print("text rel err", float(np.linalg.norm(c_w - c_c) / np.linalg.norm(c_c)), flush=True)
    zw = step(den.eps_fn(c_w, u_w, spec.guidance), z, 0, spec, 0)
    zc = step(cpu.eps_fn(c_c, u_c, spec.guidance), z, 0, spec, 0)
    print("step rel err (webgpu vs cpu)", float(np.linalg.norm(zw - zc) / np.linalg.norm(zc - z)), flush=True)
    print("browser ms per guided UNet call", timings[-3:], flush=True)


def task_debug(den: BrowserDenoiser):
    """Compare raw browser UNet output with PyTorch CPU FP32 and ONNX Runtime CPU FP16."""
    import onnxruntime as ort
    from podlab.model import Backend, Denoiser
    spec = SamplerSpec()
    z = initial_noise(0, spec)
    cpu = Denoiser(Backend("cpu", "fp32"))
    embs = np.stack([cpu.encode(""), cpu.encode("a red bicycle leaning against a brick wall")])
    sess = ort.InferenceSession(str(MODELS / "unet_fp16_torch.onnx"), providers=["CPUExecutionProvider"])
    rows = []
    for t in (961, 481, 41):
        ref = cpu.unet_eps(z, t, embs)
        o16 = sess.run(None, {"sample": np.repeat(z[None], 2, 0), "timestep": np.array([t, t], dtype=np.int64),
                              "encoder_hidden_states": embs})[0]
        web = den.unet_eps(z, t, embs)
        rel = lambda a: float(np.linalg.norm(a - ref) / np.linalg.norm(ref))
        print(f"t={t} ort-cpu-fp16 rel {rel(o16):.2e} | browser rel {rel(web):.2e} | browser finite {bool(np.isfinite(web).all())}"
              f" std {web.std():.3f} vs ref std {ref.std():.3f} | per-batch rel "
              f"{[float(np.linalg.norm(web[b] - ref[b]) / np.linalg.norm(ref[b])) for b in range(2)]}", flush=True)
        rows.append({"t": t, "ort_cpu_fp16_rel": rel(o16), "browser_rel": rel(web), "browser_finite": bool(np.isfinite(web).all())})
    # one full guided step, relative to the step size (the PoD statistic)
    zw = step(den.eps_fn(embs[1], embs[0], spec.guidance), z, 0, spec, 0)
    zc = step(cpu.eps_fn(embs[1], embs[0], spec.guidance), z, 0, spec, 0)
    r_step = float(np.linalg.norm(zw - zc) / np.linalg.norm(zc - z))
    print("browser guided-step r vs cpu fp32:", r_step, "| ms per call", timings[-4:], flush=True)
    out = ROOT / "runs" / f"browser_debug_{page_info.get('unet', 'unknown').replace('.onnx', '')}.json"
    out.write_text(json.dumps({"unet": page_info.get("unet"), "rows": rows, "guided_step_r_vs_cpu_fp32": r_step,
                               "ms_per_call": timings[-4:]}, indent=2))
    print("wrote", out, flush=True)


def task_pilot(den: BrowserDenoiser, prompts: int):
    sys.argv = [sys.argv[0]]
    import scripts.pilot as P
    from podlab.verify import check_steps
    from podlab.workers import honest
    for p in range(prompts):
        path = P.TRAJ / f"{den.name}__honest__{p:02d}.npz"
        if not path.exists():
            t0 = time.time()
            states = honest(den, den.encode(P.PROMPTS[p]), den.encode(P.SAFETY_NEG), P.seed_of(p), P.SPEC)
            P.save(path, states, time.time() - t0)
            print("webgpu honest", p, f"{time.time() - t0:.1f}s", flush=True)
    done = P.done_checks()
    for path in sorted(P.TRAJ.glob("*.npz")):
        if (den.name, path.name) in done:
            continue
        worker, kind, p = path.stem.split("__")
        p = int(p)
        states = np.load(path)["states"]
        t0 = time.time()
        checks = check_steps(den, den.encode(P.PROMPTS[p]), den.encode(P.SAFETY_NEG), states, P.CPU_STEPS, P.SPEC, P.seed_of(p))
        rec = {"verifier": den.name, "file": path.name, "worker": worker, "kind": kind, "prompt": p,
               "secs": round(time.time() - t0, 2), "checks": [c.to_dict() for c in checks]}
        with P.CHECKS.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print("webgpu verify", path.name, f"{rec['secs']}s", flush=True)


def task_study(den: BrowserDenoiser, k: int = 2):
    """Browser role in the overnight study: honest worker for each generated block, then a
    verifier with k seeded random steps on every trajectory of that block."""
    sys.argv = [sys.argv[0], "--role", "mps"]
    import scripts.study as S
    from podlab.verify import check_steps
    from podlab.workers import honest
    n = len(S.PROMPTS)
    blocks = [(b, list(range(b * S.BLOCK, min((b + 1) * S.BLOCK, n)))) for b in range((n + S.BLOCK - 1) // S.BLOCK)]
    for b, prompts in blocks:
        while not S.marker(b).exists():
            time.sleep(60)
        for p in prompts:
            path = S.TRAJ / f"{den.name}__honest__{p:02d}.npz"
            if path.exists():
                continue
            t0 = time.time()
            states = honest(den, den.encode(S.PROMPTS[p]), den.encode(S.SAFETY_NEG), S.seed_of(p), S.SPEC)
            S.save(path, states, time.time() - t0)
            print("browser gen", p, f"{time.time() - t0:.1f}s", flush=True)
        # The browser is an FP32-class verifier: check honest and FP32-class trajectories only.
        files = [f for f in S.block_files(prompts) if f.stem.split("__")[1] == "honest" or f.stem.startswith("mps-fp32__")]
        S.verify_files(den, den.name, files, all_steps_for_honest=False, k=k)
        print("== browser block done", b, flush=True)


def task_storybook(den: BrowserDenoiser):
    """Verify every page of both edge nodes in this browser (k random steps per page), page by
    page, so storybook.html can show seals as they land. Resumes: finished pages are skipped."""
    sys.argv = [sys.argv[0]]
    import scripts.storybook_case as SB
    SB.OUT = STORY  # which rendered edition to verify (--story-run)
    book = json.loads((STORY / "book.json").read_text())
    taus = SB.load_taus(den.name)
    out = STORY / f"verify_{den.name}.jsonl"
    done = set()
    if out.exists():
        done = {(r["node"], r["page"]) for r in map(json.loads, out.read_text().splitlines()) if r}
    for page in book["pages"]:
        for node in ("A", "B"):
            if (node, page["id"]) in done:
                continue
            story_current["job"] = {"page": page["id"], "node": node}
            rec = SB.verify_page(den, den.name, book, page, node, taus)
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print("storybook", den.name, node, page["id"], "ACCEPT" if rec["accepted"] else "REJECT", f"{rec['secs']}s", flush=True)
    story_current["job"] = None
    print("storybook verification finished", flush=True)


CALIB = {"runs": ["study"], "prompts": 8}


def task_calibrate(den: BrowserDenoiser):
    """Per-verifier calibration for the browser: check ALL steps of honest FP32 trajectories made
    on other devices (study prompts, never the storybook test pages). Sources: CALIB["runs"]
    (runs/<name>/traj + spec.json), prompts < CALIB["prompts"]. Output: runs/calib_<name>/."""
    sys.argv = [sys.argv[0]]
    from podlab.verify import check_steps
    out = ROOT / "runs" / f"calib_{den.name}"
    out.mkdir(parents=True, exist_ok=True)
    f_out = out / "checks.jsonl"
    done = {r["file"] for r in map(json.loads, f_out.read_text().splitlines()) if r} if f_out.exists() else set()
    for run in CALIB["runs"]:
        src = ROOT / "runs" / run
        if not (src / "spec.json").exists():
            print("calibrate: no run", run, flush=True)
            continue
        spec_doc = json.loads((src / "spec.json").read_text())
        spec = SamplerSpec(**spec_doc["spec"])
        files = sorted(p for p in (src / "traj").glob("*-fp32__honest__*.npz")
                       if not p.name.startswith("webgpu") and int(p.stem.split("__")[2]) < CALIB["prompts"])
        for path in files:
            key = path.name if run == "study" else f"{run}/{path.name}"
            if key in done:
                continue
            worker, kind, p = path.stem.split("__")
            p = int(p)
            states = np.load(path)["states"]
            t0 = time.time()
            checks = check_steps(den, den.encode(spec_doc["prompts"][p]), den.encode(spec_doc.get("safety_neg", "")),
                                 states, list(range(spec.steps)), spec, spec_doc.get("seed_base", 5000) + p)
            rec = {"verifier": den.name, "file": key, "worker": worker, "kind": kind, "prompt": p, "run": run,
                   "secs": round(time.time() - t0, 2), "checks": [c.to_dict() for c in checks]}
            with f_out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print("calibrate", den.name, key, f"{rec['secs']}s", flush=True)
    print("calibration finished", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--task", default="smoke", choices=["smoke", "pilot", "serve", "debug", "study", "storybook", "calibrate"])
    ap.add_argument("--prompts", type=int, default=12)
    ap.add_argument("--calib-runs", default="study,nvidia", help="runs whose honest FP32 trajectories calibrate the browser")
    ap.add_argument("--calib-prompts", type=int, default=8)
    ap.add_argument("--story-run", default="storybook", help="runs/<name> holding the edition to verify (e.g. storybook_cuda)")
    a = ap.parse_args()
    global STORY
    STORY = ROOT / "runs" / a.story_run
    CALIB.update(runs=a.calib_runs.split(","), prompts=a.calib_prompts)
    if a.task in ("storybook", "calibrate"):  # pin the FP32 graph (FP16 WebGPU is too noisy)
        ROOT_REDIRECT["to"] = "/storybook?unet=unet_fp32_single.onnx&ext=1"
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"serving http://127.0.0.1:{a.port}/ ; open it in a WebGPU browser", flush=True)
    ready.wait()
    den = BrowserDenoiser()
    if a.task == "smoke":
        task_smoke(den)
    elif a.task == "pilot":
        task_pilot(den, a.prompts)
    elif a.task == "debug":
        task_debug(den)
    elif a.task == "study":
        task_study(den)
    elif a.task == "storybook":
        task_storybook(den)
        threading.Event().wait()  # keep serving the reader page
    elif a.task == "calibrate":
        task_calibrate(den)
    if a.task != "serve":
        jobs.put({"id": 0, "op": "stop"})
        time.sleep(1)
    else:
        threading.Event().wait()


if __name__ == "__main__":
    main()
