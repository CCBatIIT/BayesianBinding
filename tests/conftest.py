"""Pytest configuration for bayesian_binding tests.

Expose a few CPU cores to JAX as separate devices before it is imported, so the
regression tests can sample a handful of NUTS chains in parallel. This must run
before any jax/numpyro import, which pytest guarantees by importing conftest
first.

Four devices is deliberate: for these small, tree-heavy ITC models NUTS is
latency-bound per chain, and CPU ``pmap`` over many devices adds overhead plus a
"slowest chain gates each step" straggler effect, so more chains run *slower*, not
faster. Wall time is reduced by using fewer posterior samples (see the regression
tests), not more chains.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
# Use a non-interactive matplotlib backend so plotting helpers import/run headless under pytest.
os.environ.setdefault("MPLBACKEND", "Agg")
