"""Small plotting helpers for notebooks and reports."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from bayesian_binding.data import ITCExperiment


def plot_fit(experiment: ITCExperiment, q_model, ax=None, *, title: str | None = None):
    """Plot observed heats and one posterior/model prediction curve."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 3.5))
    x = np.arange(1, experiment.n_injections + 1)
    ax.scatter(x, experiment.heats_microcalorie, color="black", s=24, label="observed")
    ax.plot(x, np.asarray(q_model), color="#2f6f9f", lw=2, label="model")
    ax.axhline(0.0, color="0.82", lw=0.8)
    ax.set_xlabel("injection")
    ax.set_ylabel("heat (microcal)")
    ax.set_title(title or experiment.name)
    ax.legend(frameon=False)
    return ax

