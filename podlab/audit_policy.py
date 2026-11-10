"""Conservative planning tools for a sampled diffusion audit.

These helpers are a standard split-conformal construction and an exact sampling
calculation, not evidence of security against arbitrary adversaries. They do not
change the historical experiment thresholds or certify the delivered image.

Calibration uses ONE maximum score per independent request group, never treats
the correlated denoising steps as independent observations. Groups must be
exchangeable with future honest request groups, with the same model, precision,
backend policy and step layout. Repeated seeds/prompts that share a dependence
must be combined into one group by the caller. No statistical API can infer that
independence from an identifier alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Integral
from statistics import median
from typing import Mapping, Sequence


@dataclass(frozen=True)
class DriftGroup:
    """Per-step maximum residuals over all dependent runs in a request group."""

    group_id: str
    ratios: tuple[float, ...]


@dataclass(frozen=True)
class Calibration:
    alpha: float
    n_groups: int
    rank: int
    scales: tuple[float, ...]
    thresholds: tuple[float, ...] | None
    fit_ids: tuple[str, ...]
    calibration_ids: tuple[str, ...]

    @property
    def usable(self) -> bool:
        return self.thresholds is not None

    def assess(self, observed: Mapping[int, float]) -> str:
        """Return inconclusive, flagged, or within_threshold for numerical checks.

        Empty, invalid or too-small calibrations never produce a passing result.
        Merkle/header/challenge validation remains the protocol's responsibility.
        A within_threshold result concerns only the supplied sampled residuals.
        """
        if not observed:
            return "inconclusive"
        for index, ratio in observed.items():
            if (not isinstance(index, Integral) or isinstance(index, bool)
                    or not 0 <= index < len(self.scales)):
                raise ValueError("observed step index is outside the calibrated layout")
            if not math.isfinite(ratio) or ratio < 0:
                raise ValueError("residuals must be finite and nonnegative")
        if self.thresholds is None:
            return "inconclusive"
        return ("flagged" if any(r > self.thresholds[i] for i, r in observed.items())
                else "within_threshold")


def _probability(value: float, name: str, *, include_one: bool = False) -> Fraction:
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite probability")
    result = Fraction(str(value))
    if not (0 < result <= 1 if include_one else 0 < result < 1):
        raise ValueError(f"{name} must be in (0, {'1]' if include_one else '1)'}")
    return result


def _count(value: int, name: str, minimum: int = 0) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _validate_groups(groups: Sequence[DriftGroup], steps: int) -> tuple[str, ...]:
    ids = []
    for group in groups:
        if not isinstance(group.group_id, str) or not group.group_id.strip():
            raise ValueError("every group requires a nonempty identifier")
        if len(group.ratios) != steps:
            raise ValueError("each group must include every step in the same layout")
        if any(not math.isfinite(r) or r < 0 for r in group.ratios):
            raise ValueError("residuals must be finite and nonnegative")
        ids.append(group.group_id)
    if len(set(ids)) != len(ids):
        raise ValueError("dependent repeats must be aggregated into one group")
    return tuple(ids)


def calibrate_request_max(
    calibration: Sequence[DriftGroup], *, steps: int, alpha: float = 0.01,
    scale_fit: Sequence[DriftGroup] = (),
) -> Calibration:
    """Fit fixed scales on a disjoint split, then calibrate whole-request maxima.

    With scales a_i fixed before calibration, use S_j=max_i r_ji/a_i, and the
    ceil((n+1)(1-alpha))-th order statistic q of the n group scores. Thresholds
    are q*a_i. Under group exchangeability, an honest group's probability of
    ANY exceedance is <= alpha; any subset of checked steps inherits this bound.
    This is marginal coverage, not a conditional guarantee for every prompt.

    A rank of n+1 needs an infinite threshold and has no detection power. Return
    thresholds=None instead so a caller must report insufficient calibration.
    At alpha=.01 at least 99 calibration groups are required (in addition to
    scale-fit groups). This condition is necessary, not sufficient for power.

    Default scales are all 1 (no fitting). Optional scales are stepwise medians
    from scale_fit only. Neither thresholds nor scale selection may be tuned
    against the evaluation set without restarting the holdout procedure.
    """
    steps = _count(steps, "steps", 1)
    alpha_fraction = _probability(alpha, "alpha")
    fit_ids = _validate_groups(scale_fit, steps)
    calibration_ids = _validate_groups(calibration, steps)
    if set(fit_ids) & set(calibration_ids):
        raise ValueError("scale-fit and calibration group IDs must be disjoint")
    scales = tuple(max(float(median(g.ratios[i] for g in scale_fit)), 1e-12)
                   if scale_fit else 1.0 for i in range(steps))
    scores = sorted(max(r / a for r, a in zip(g.ratios, scales)) for g in calibration)
    if any(not math.isfinite(s) for s in scores):
        raise ValueError("normalized calibration residuals overflowed")
    rank = math.ceil((len(scores) + 1) * (1 - alpha_fraction))
    thresholds = None
    if rank <= len(scores):
        thresholds = tuple(scores[rank - 1] * a for a in scales)
        if any(not math.isfinite(t) for t in thresholds):
            raise ValueError("calibrated thresholds overflowed")
    return Calibration(float(alpha_fraction), len(scores), rank, scales, thresholds,
                       fit_ids, calibration_ids)


def exact_detection_probability(steps: int, detectable_steps: int, checks: int) -> Fraction:
    """Exact chance of hitting a detectable step, conditional on its count.

    Requires distinct uniform challenges drawn AFTER the root is fixed. A step
    that merely saves compute is not necessarily detectable at the threshold.
    """
    steps = _count(steps, "steps", 1)
    detectable_steps = _count(detectable_steps, "detectable_steps")
    checks = _count(checks, "checks")
    if detectable_steps > steps or checks > steps:
        raise ValueError("detectable_steps and checks cannot exceed steps")
    return 1 - Fraction(math.comb(steps - detectable_steps, checks), math.comb(steps, checks))


@dataclass(frozen=True)
class AuditBudget:
    steps: int
    min_detectable_steps: int
    target: float
    checks: int | None
    achieved: float

    @property
    def network_evaluation_fraction(self) -> float | None:
        """Step count fraction, NOT total latency/energy/monetary overhead."""
        return None if self.checks is None else self.checks / self.steps


def plan_audit(steps: int, min_detectable_steps: int, target: float = 0.99) -> AuditBudget:
    """Smallest uniform sample meeting target for an explicit detectable-step bound.

    The bound is a threat-model assumption. Never infer it from the requested
    quality, from a compute-saving percentage, or from an untrusted worker.
    No positive target is attainable if min_detectable_steps is zero.
    """
    target_fraction = _probability(target, "target", include_one=True)
    exact_detection_probability(steps, min_detectable_steps, 0)  # validate counts
    for k in range(1, steps + 1):
        probability = exact_detection_probability(steps, min_detectable_steps, k)
        if probability >= target_fraction:
            return AuditBudget(steps, min_detectable_steps, float(target_fraction), k, float(probability))
    return AuditBudget(steps, min_detectable_steps, float(target_fraction), None, 0.0)
