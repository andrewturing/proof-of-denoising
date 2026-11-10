"""Output folders for figures and LaTeX tables.

Inside the research workspace (a sibling ``poster/`` or ``paper-pod/`` folder exists), scripts write there, as they
always have. In a standalone checkout of this repository they write to ``figures/poster`` and ``figures/paper``
inside the repository instead, never outside it.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _dir(sibling: Path, fallback: Path) -> Path:
    d = sibling if sibling.is_dir() else fallback
    d.mkdir(parents=True, exist_ok=True)
    return d


def poster_figs() -> Path:
    """Poster figures (PNG): ../poster/figs, or figures/poster."""
    return _dir(ROOT.parent / "poster" / "figs", ROOT / "figures" / "poster")


def paper_dir() -> Path:
    """Paper root holding fig/, tables/ and the results macros: ../paper-pod, or figures/paper."""
    d = _dir(ROOT.parent / "paper-pod", ROOT / "figures" / "paper")
    for sub in ("fig", "tables"):
        (d / sub).mkdir(exist_ok=True)
    return d
