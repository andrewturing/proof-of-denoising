// Proof-of-Denoising WebGPU worker: runs the SD1.5 UNet and text encoder with ONNX Runtime Web
// for jobs handed out by the local harness (scripts/browser_backend.py). Shared by index.html
// (plain worker) and storybook.html (reader that verifies its own pages).
import * as ort from '/vendor/ort.webgpu.bundle.min.mjs';

ort.env.wasm.wasmPaths = '/vendor/';
ort.env.wasm.numThreads = 1;

function b64ToF32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

export async function startWorker({ log = () => {}, onStage = () => {} } = {}) {
  const say = (msg) => {
    log(msg);
    fetch('/log', { method: 'POST', body: msg }).catch(() => {});
  };
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) { say('ERROR: no WebGPU adapter'); onStage({ stage: 'error', detail: 'WebGPU is not available' }); return; }
  const info = { vendor: adapter.info?.vendor, arch: adapter.info?.architecture, f16: adapter.features.has('shader-f16') };
  say('adapter ' + JSON.stringify(info));
  onStage({ stage: 'adapter', detail: info });

  const q = new URLSearchParams(location.search);
  const opts = { executionProviders: [q.get('ep') || 'webgpu'], graphOptimizationLevel: q.get('opt') || 'all' };
  say('session options ' + JSON.stringify(opts));
  const unetFile = q.get('unet') || 'unet_fp16_torch.onnx';
  const unetOpts = { ...opts };
  if (q.get('ext')) {
    // FP32 weights live in one external-data file next to the graph.
    unetOpts.externalData = [{ path: unetFile + '.data', data: '/models/' + unetFile + '.data' }];
  }
  onStage({ stage: 'loading', detail: unetFile });
  let t0 = performance.now();
  const unet = await ort.InferenceSession.create('/models/' + unetFile, unetOpts);
  say('unet file ' + unetFile);
  say(`unet loaded in ${Math.round(performance.now() - t0)} ms`);
  t0 = performance.now();
  const text = await ort.InferenceSession.create('/models/text_encoder_fp32.onnx', opts);
  say(`text encoder loaded in ${Math.round(performance.now() - t0)} ms`);
  onStage({ stage: 'ready', detail: unetFile });
  await fetch('/ready', { method: 'POST' });

  for (;;) {
    const r = await fetch('/next');
    if (r.status === 204) continue;
    const job = await r.json();
    if (job.op === 'stop') { say('stopped'); onStage({ stage: 'stopped' }); return; }
    const t1 = performance.now();
    let out;
    try {
      if (job.op === 'unet') {
        const feeds = {
          sample: new ort.Tensor('float32', b64ToF32(job.sample), [2, 4, 64, 64]),
          timestep: new ort.Tensor('int64', BigInt64Array.from([BigInt(job.t), BigInt(job.t)]), [2]),
          encoder_hidden_states: new ort.Tensor('float32', b64ToF32(job.embs), [2, 77, 768]),
        };
        out = (await unet.run(feeds)).eps.data;
      } else if (job.op === 'text') {
        const ids = new ort.Tensor('int64', BigInt64Array.from(job.ids.map(BigInt)), [1, 77]);
        out = (await text.run({ input_ids: ids })).last_hidden_state.data;
      }
    } catch (e) {
      say('ERROR ' + e);
      await fetch(`/result/${job.id}?error=1`, { method: 'POST', body: String(e) });
      continue;
    }
    const ms = Math.round(performance.now() - t1);
    onStage({ stage: 'computed', detail: { op: job.op, ms } });
    await fetch(`/result/${job.id}?ms=${ms}`, { method: 'POST', body: new Float32Array(out).buffer });
  }
}
