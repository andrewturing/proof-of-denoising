"""Protocol regressions against malformed and strategically incomplete responses.

Uses a deterministic stand-in, not downloaded model weights. Numerical backend
drift is covered by the experiment logs, not these security boundary tests.
"""
import itertools
import unittest
from dataclasses import replace

import numpy as np

from podlab.protocol import Job, draw_challenges, open_steps, run_and_commit, verify_openings
from podlab.sampler import SamplerSpec, initial_noise
from podlab.commit import commit
from tests.test_workers import FakeDenoiser

SPEC = SamplerSpec(steps=10, height=32, width=32)
JOB = Job(prompt="a cat", negative="unsafe", seed=3, spec=SPEC)
TAUS = {i: 1e-6 for i in range(SPEC.steps)}


class ProtocolTest(unittest.TestCase):
    def setUp(self):
        self.states, self.tree = run_and_commit(FakeDenoiser(), JOB)

    def verify(self, idx, *, states=None, tree=None, openings=None, taus=None, **overrides):
        states = self.states if states is None else states
        tree = self.tree if tree is None else tree
        args = dict(challenges=idx, header_path=tree.path(0), initial_state=states[0],
                    initial_path=tree.path(1))
        args.update(overrides)
        return verify_openings(FakeDenoiser(), JOB, tree.root,
                               open_steps(states, tree, idx) if openings is None else openings,
                               TAUS if taus is None else taus, **args)

    def test_honest_worker_is_accepted(self):
        idx = draw_challenges(SPEC.steps, 4)
        ok, checks = self.verify(idx)
        self.assertTrue(ok)
        self.assertEqual([c.i for c in checks], idx)

    def test_honest_response_can_arrive_out_of_order(self):
        idx = [2, 7]
        ok, _ = self.verify(idx, openings=list(reversed(open_steps(self.states, self.tree, idx))))
        self.assertTrue(ok)

    def test_substituted_model_is_rejected(self):
        states, tree = run_and_commit(FakeDenoiser(bias=0.05), JOB)
        self.assertFalse(self.verify([3], states=states, tree=tree)[0])

    def test_empty_incomplete_duplicate_or_substituted_responses_are_rejected(self):
        idx = [2, 7]
        for response_indices in ([], [2], [2, 2], [2, 3], [2, 7, 8]):
            with self.subTest(response_indices=response_indices):
                self.assertFalse(self.verify(idx, openings=open_steps(self.states, self.tree, response_indices))[0])

    def test_invalid_verifier_challenges_are_rejected(self):
        for idx in ([], [2, 2], [-1], [SPEC.steps], [True], [1.0]):
            with self.subTest(idx=idx):
                self.assertFalse(self.verify(idx, openings=[])[0])

    def test_tampered_opening_fails_before_recomputation(self):
        openings = open_steps(self.states, self.tree, [2])
        openings[0].z_next = openings[0].z_next + 1e-3
        ok, checks = self.verify([2], openings=openings)
        self.assertFalse(ok)
        self.assertEqual(checks, [])

    def test_job_header_is_always_bound(self):
        tree = commit(replace(JOB, model_id="unrelated-weights").header(), self.states)
        self.assertFalse(self.verify([3, 7], tree=tree)[0])
        self.assertFalse(self.verify([3, 7], header_path=None)[0])

    def test_wrong_seed_is_rejected_without_challenging_first_step(self):
        # With deterministic DDIM every transition of this wrong-seed trajectory
        # is locally valid. A mandatory initial-state proof closes that bypass.
        states, _ = run_and_commit(FakeDenoiser(), replace(JOB, seed=4))
        tree = commit(JOB.header(), states)
        self.assertFalse(self.verify([3, 7], states=states, tree=tree)[0])
        self.assertFalse(self.verify([3, 7], states=states, tree=tree,
                                     initial_state=initial_noise(JOB.seed, SPEC))[0])

    def test_float64_hash_alias_and_broadcasting_are_rejected(self):
        for convert in (lambda x: x.astype(np.float64), lambda x: x[:1]):
            openings = open_steps(self.states, self.tree, [2])
            openings[0].z_i = convert(openings[0].z_i)
            self.assertFalse(self.verify([2], openings=openings)[0])

    def test_nonfinite_committed_state_is_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            states = self.states.copy()
            states[3, 0, 0, 0] = value
            tree = commit(JOB.header(), states)
            self.assertFalse(self.verify([2], states=states, tree=tree)[0])

    def test_missing_nonfinite_or_negative_threshold_is_rejected(self):
        for value in (None, np.nan, np.inf, -1, True):
            taus = {} if value is None else {2: value}
            self.assertFalse(self.verify([2], taus=taus)[0])

    def test_wrong_path_position_or_length_is_rejected(self):
        for path in (self.tree.path(1), self.tree.path(0)[:-1], self.tree.path(0) + self.tree.path(0)[:1]):
            self.assertFalse(self.verify([2], header_path=path)[0])

    def test_sparse_splice_exposes_the_sampling_limit(self):
        # A valid suffix from another seed, spliced after the true initial noise,
        # has only one invalid transition. All other local checks must pass.
        states, _ = run_and_commit(FakeDenoiser(), replace(JOB, seed=4))
        states[0] = initial_noise(JOB.seed, SPEC)
        tree = commit(JOB.header(), states)
        pairs = list(itertools.combinations(range(SPEC.steps), 2))
        detected = sum(not self.verify(list(idx), states=states, tree=tree)[0] for idx in pairs)
        self.assertAlmostEqual(detected / len(pairs), 2 / SPEC.steps)

    def test_challenges_are_distinct_and_reproducible_only_when_explicitly_seeded(self):
        idx = draw_challenges(25, 8)
        self.assertEqual(len(set(idx)), 8)
        self.assertTrue(all(0 <= i < 25 for i in idx))
        self.assertEqual(draw_challenges(25, 3, np.random.default_rng(1)),
                         draw_challenges(25, 3, np.random.default_rng(1)))
        for steps, k in ((25, 0), (25, -1), (25, 26), (0, 1), (25, True)):
            with self.assertRaises(ValueError):
                draw_challenges(steps, k)


if __name__ == "__main__":
    unittest.main()
