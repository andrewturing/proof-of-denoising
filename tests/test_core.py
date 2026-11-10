import unittest

import numpy as np

from podlab.commit import MerkleTree, commit, header_leaf, state_leaf, verify_path, verify_indexed_path
from podlab.sampler import SamplerSpec, initial_noise, sample, step_noise
from podlab.verify import detection_probability


def fake_eps(z, t):
    # Deterministic, nonlinear stand-in for a UNet.
    return np.tanh(0.3 * z + 0.001 * t).astype(np.float32)


class SamplerTest(unittest.TestCase):
    def test_matches_diffusers_ddim(self):
        import torch
        from diffusers import DDIMScheduler

        spec = SamplerSpec(steps=10, height=64, width=64)
        sched = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                              num_train_timesteps=1000, steps_offset=1, set_alpha_to_one=False,
                              clip_sample=False, timestep_spacing="leading")
        sched.set_timesteps(spec.steps)
        self.assertEqual(list(sched.timesteps.numpy()), list(spec.timesteps()))
        ours = sample(fake_eps, seed=7, spec=spec)
        z = torch.from_numpy(initial_noise(7, spec).astype(np.float64))
        for i, t in enumerate(sched.timesteps):
            eps = torch.from_numpy(fake_eps(z.float().numpy(), int(t)).astype(np.float64))
            z = sched.step(eps, t, z, eta=0.0).prev_sample
            np.testing.assert_allclose(ours[i + 1], z.numpy(), rtol=0, atol=1e-5)

    def test_noise_is_reproducible_and_step_specific(self):
        spec = SamplerSpec(height=64, width=64)
        np.testing.assert_array_equal(initial_noise(3, spec), initial_noise(3, spec))
        self.assertFalse(np.array_equal(step_noise(3, 0, spec), step_noise(3, 1, spec)))
        self.assertFalse(np.array_equal(initial_noise(3, spec), initial_noise(4, spec)))

    def test_stochastic_sampler_is_markov_given_seed(self):
        spec = SamplerSpec(steps=6, eta=1.0, height=64, width=64)
        a = sample(fake_eps, 11, spec)
        b = sample(fake_eps, 11, spec)
        np.testing.assert_array_equal(a, b)


class CommitTest(unittest.TestCase):
    def test_indexed_paths_bind_position_size_and_padding(self):
        for n in range(1, 18):
            leaves = [header_leaf({"i": i}) for i in range(n)]
            tree = MerkleTree.build(leaves)
            for i, leaf in enumerate(leaves):
                path = tree.path(i)
                self.assertTrue(verify_indexed_path(leaf, path, tree.root, i, n))
                self.assertFalse(verify_indexed_path(leaf, path, tree.root, n, n))
                if n > 1:
                    self.assertFalse(verify_indexed_path(leaf, path, tree.root, (i + 1) % n, n))
                    self.assertFalse(verify_indexed_path(leaf, path[:-1], tree.root, i, n))
            with self.assertRaises(ValueError):
                tree.path(-1)
            with self.assertRaises(ValueError):
                tree.path(n)
        with self.assertRaises(ValueError):
            MerkleTree.build([])

    def test_paths_verify_and_tampering_fails(self):
        states = np.random.default_rng(0).standard_normal((7, 4, 8, 8)).astype(np.float32)
        header = {"model": "x", "seed": 1}
        tree = commit(header, states)
        self.assertTrue(verify_path(header_leaf(header), tree.path(0), tree.root))
        for j in range(len(states)):
            self.assertTrue(verify_path(state_leaf(j, states[j]), tree.path(j + 1), tree.root))
        bad = states[3].copy()
        bad[0, 0, 0] = np.nextafter(bad[0, 0, 0], np.float32(np.inf))
        self.assertFalse(verify_path(state_leaf(3, bad), tree.path(4), tree.root))
        self.assertFalse(verify_path(state_leaf(2, states[3]), tree.path(4), tree.root))


class DetectionTest(unittest.TestCase):
    def test_detection_probability(self):
        self.assertAlmostEqual(detection_probability(25, 25, 1), 1.0)
        self.assertAlmostEqual(detection_probability(0, 25, 4), 0.0)
        self.assertAlmostEqual(detection_probability(12, 25, 1), 12 / 25)
        self.assertGreater(detection_probability(12, 25, 4), 1 - (13 / 25) ** 4)


if __name__ == "__main__":
    unittest.main()
