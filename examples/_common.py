"""Shared helpers for the BayesianBinding example fitting scripts.

Each example under ``examples/`` is split into a fitting script (``fit.py``), which runs a
MAP fit and MCMC and saves the posterior samples, and a companion analysis notebook
(``analysis.ipynb``), which loads those saved samples and produces tables and figures. This
module holds the small pieces both halves share: locating the repository data, the
``--quick`` / ``--full`` command-line scaffolding, and reading/writing the saved-results
bundle (``posterior_samples.npz`` + ``posterior_summary.csv`` + ``run_metadata.json``).
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd

# examples/_common.py -> examples/ -> repository root.
REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"


@dataclass(frozen=True)
class RunSettings:
    """NUTS settings for one run mode (``--quick`` or ``--full``)."""

    num_warmup: int
    num_samples: int
    num_chains: int


def add_run_arguments(parser, *, default_out: Path, default_seed: int = 20260613):
    """Add the shared ``--quick``/``--full``/``--seed``/``--out`` options to ``parser``."""
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        dest="mode",
        action="store_const",
        const="quick",
        help="Short run to check the code works (seconds to ~1 min).",
    )
    mode.add_argument(
        "--full",
        dest="mode",
        action="store_const",
        const="full",
        help="Long run that reproduces the published result and stores the MCMC samples.",
    )
    parser.set_defaults(mode="quick")
    parser.add_argument("--seed", type=int, default=default_seed, help="NUTS PRNG seed.")
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help="Directory for saved samples, summary, and metadata.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable the NUTS progress bar.")
    return parser


def posterior_parameter_summary(idata: az.InferenceData) -> pd.DataFrame:
    """ArviZ posterior summary excluding the deterministic ``q_model*`` heat vectors."""
    var_names = [v for v in idata.posterior.data_vars if not str(v).startswith("q_model")]
    return az.summary(idata, var_names=var_names)


def run_environment() -> dict[str, str]:
    """Record the Python/OS environment and a UTC timestamp for reproducibility."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def save_results(
    out_dir: Path,
    *,
    samples: dict[str, Any],
    idata: az.InferenceData,
    metadata: dict[str, Any],
) -> Path:
    """Write the posterior samples, parameter summary, and run metadata to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "posterior_samples.npz",
        **{name: np.asarray(value) for name, value in samples.items()},
    )
    posterior_parameter_summary(idata).to_csv(out_dir / "posterior_summary.csv")
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    return out_dir


def load_results(out_dir: Path) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    """Load a saved-results bundle written by :func:`save_results`.

    Returns ``(samples, summary, metadata)``: the posterior samples as a dict of arrays, the
    ArviZ parameter summary as a DataFrame, and the run metadata as a dict.
    """
    out_dir = Path(out_dir)
    if not (out_dir / "posterior_samples.npz").exists():
        raise FileNotFoundError(
            f"No saved samples in {out_dir}. Run the example's fit.py (e.g. `--quick`) first."
        )
    with np.load(out_dir / "posterior_samples.npz") as handle:
        samples = {name: handle[name] for name in handle.files}
    summary = pd.read_csv(out_dir / "posterior_summary.csv", index_col=0)
    metadata = json.loads((out_dir / "run_metadata.json").read_text())
    return samples, summary, metadata


def mcmc_metadata(settings: RunSettings) -> dict[str, int]:
    """Return the NUTS settings as a plain dict for run metadata."""
    return asdict(settings)
