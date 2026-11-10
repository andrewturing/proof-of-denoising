"""Worker and verifier logic with a fake, deterministic denoiser (no model weights)."""
import unittest

import numpy as np

from podlab.sampler import SamplerSpec
from podlab.verify import check_steps
from podlab.workers import cfg_skip, eps_cache, honest, neg_strip, step_skip


class FakeDenoiser:
    """Stands in for podlab.model.Denoiser: a fixed nonlinear map of (z, t, embedding)."""

    def __init__(self, bias: float = 0.0):
        self.bias = bias

    def encode(self, prompt: str) -> np.ndarray:
        return np.full((2, 3), (len(prompt) % 7) / 7.0, dtype=np.float32)

    def unet_eps(self, z, t, embs):
        return np.stack([np.tanh(0.3 * z + 0.001 * t + e.mean() + self.bias) for e in embs]).astype(np.float32)

    def eps_fn(self, cond, uncond, guidance):
        embs = np.stack([uncond, cond])

        def f(z, t):
            e_u, e_c = self.unet_eps(z, t, embs)
            return e_u + guidance * (e_c - e_u)

        return f


SPEC = SamplerSpec(steps=10, height=32, width=32)


def run(worker, den, **kw):
    return worker(den, den.encode("a cat"), den.encode("unsafe"), 5, SPEC, **kw)


def ratios(states, den=None):
    den = den or FakeDenoiser()
    checks = check_steps(den, den.encode("a cat"), den.encode("unsafe"), states, list(range(SPEC.steps)), SPEC, 5)
    return np.array([c.ratio for c in checks])


class WorkerTest(unittest.TestCase):
    def test_honest_verifies_exactly(self):
        states = run(honest, FakeDenoiser())
        self.assertEqual(states.shape, (SPEC.steps + 1, 4, 4, 4))
        self.assertTrue(np.all(ratios(states) == 0))

    def test_substitution_deviates_on_every_step(self):
        self.assertTrue(np.all(ratios(run(honest, FakeDenoiser(bias=0.05))) > 0))

    def test_eps_cache_deviates_only_on_skipped_steps(self):
        r = ratios(run(eps_cache, FakeDenoiser(), period=2))
        self.assertTrue(np.all(r[0::2] == 0))
        self.assertTrue(np.all(r[1::2] > 0))
        late = ratios(run(eps_cache, FakeDenoiser(), period=2, start=6))
        self.assertTrue(np.all(late[:7] == 0))
        self.assertTrue(np.all(late[7::2] > 0))

    def test_step_skip_and_cfg_skip_deviate(self):
        self.assertTrue(np.all(ratios(run(step_skip, FakeDenoiser())) > 0))
        self.assertTrue(np.all(ratios(run(cfg_skip, FakeDenoiser())) > 0))

    def test_negative_prompt_stripping_deviates(self):
        den = FakeDenoiser()
        states = neg_strip(den, den.encode("a cat"), den.encode(""), 5, SPEC)
        self.assertTrue(np.all(ratios(states) > 0))


if __name__ == "__main__":
    unittest.main()
