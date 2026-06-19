"""Load integrated and raw ITC data.

The loaders use a small, explicit data model with SI-adjacent units:

- heats are microcalories;
- injection volumes are liters;
- concentrations are molar;
- cell volume is liters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.integrate import trapezoid

from bayesian_binding.constants import (
    DEFAULT_TEMPERATURE_K,
    LITERS_PER_MICROLITER,
    LITERS_PER_MILLILITER,
    MOLAR_PER_MILLIMOLAR,
)


@dataclass(frozen=True)
class RawInjectionBlock:
    """A raw MicroCal injection block."""

    number: int
    volume_liter: float
    duration_s: float | None
    points: np.ndarray


@dataclass(frozen=True)
class ITCExperiment:
    """Integrated ITC experiment ready for regression."""

    name: str
    injection_volumes_liter: np.ndarray
    heats_microcalorie: np.ndarray
    cell_concentration_molar: float
    syringe_concentration_molar: float
    cell_volume_liter: float
    temperature_k: float = DEFAULT_TEMPERATURE_K
    source_path: Path | None = None
    raw_blocks: tuple[RawInjectionBlock, ...] = ()

    @property
    def n_injections(self) -> int:
        return int(self.heats_microcalorie.size)

    def without_first_injection(self) -> "ITCExperiment":
        """Return a copy with the first injection removed."""
        return ITCExperiment(
            name=f"{self.name}_drop_first",
            injection_volumes_liter=self.injection_volumes_liter[1:],
            heats_microcalorie=self.heats_microcalorie[1:],
            cell_concentration_molar=self.cell_concentration_molar,
            syringe_concentration_molar=self.syringe_concentration_molar,
            cell_volume_liter=self.cell_volume_liter,
            temperature_k=self.temperature_k,
            source_path=self.source_path,
            raw_blocks=self.raw_blocks[1:] if self.raw_blocks else (),
        )


def _parse_float(text: str) -> float | None:
    try:
        return float(text.strip())
    except ValueError:
        return None


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def load_dat(
    path: str | Path,
    *,
    cell_concentration_mM: float | None = None,
    syringe_concentration_mM: float | None = None,
    cell_volume_mL: float = 1.434,
    temperature_k: float = DEFAULT_TEMPERATURE_K,
    name: str | None = None,
) -> ITCExperiment:
    """Load an Origin/NITPIC-style `.DAT` integrated heat file.

    The first numeric column is treated as integrated heat in microcalories and
    the second numeric column as injection volume in microliters. If concentration
    metadata is not supplied, the loader uses the first row's `Mt` and inferred
    syringe concentration from the `Xt` progression when available.
    """
    path = _as_path(path)
    rows: list[list[float]] = []
    with path.open(errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("DH"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                rows.append([float(part) for part in parts[:6]])
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No integrated heat rows found in {path}")

    data = np.asarray(rows, dtype=float)
    heats = data[:, 0]
    injection_volumes_liter = data[:, 1] * LITERS_PER_MICROLITER

    if cell_concentration_mM is None:
        cell_concentration_mM = float(data[0, 3])
    if syringe_concentration_mM is None:
        syringe_concentration_mM = _infer_syringe_concentration_mM(
            xt_mM=data[:, 2],
            injection_volumes_liter=injection_volumes_liter,
            cell_volume_liter=cell_volume_mL * 1e-3,
        )

    return ITCExperiment(
        name=name or path.stem,
        injection_volumes_liter=injection_volumes_liter,
        heats_microcalorie=heats,
        cell_concentration_molar=float(cell_concentration_mM) * MOLAR_PER_MILLIMOLAR,
        syringe_concentration_molar=float(syringe_concentration_mM) * MOLAR_PER_MILLIMOLAR,
        cell_volume_liter=float(cell_volume_mL) * 1e-3,
        temperature_k=float(temperature_k),
        source_path=path,
    )


def _infer_syringe_concentration_mM(
    xt_mM: np.ndarray,
    injection_volumes_liter: np.ndarray,
    cell_volume_liter: float,
) -> float:
    """Infer syringe concentration from the first nonzero Xt increment."""
    for index in range(1, len(xt_mM)):
        delta_xt = xt_mM[index] - xt_mM[index - 1] * (1.0 - injection_volumes_liter[index] / cell_volume_liter)
        if delta_xt > 0:
            return float(delta_xt * cell_volume_liter / injection_volumes_liter[index])
    raise ValueError("Could not infer syringe concentration; pass syringe_concentration_mM explicitly.")


def parse_itc_raw(path: str | Path) -> tuple[dict[str, float], tuple[RawInjectionBlock, ...]]:
    """Parse a MicroCal-style `.itc` text file into metadata and raw blocks."""
    path = _as_path(path)
    dollar_values: list[str] = []
    hash_values: list[float | None] = []
    injection_specs: list[tuple[float, float | None]] = []
    blocks: dict[int, tuple[float, float | None, list[tuple[float, float, float]]]] = {}
    current_block: int | None = None

    with path.open(errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("$"):
                value = line[1:].strip()
                dollar_values.append(value)
                fields = [field.strip() for field in value.split(",")]
                if len(fields) >= 1:
                    first = _parse_float(fields[0])
                    if first is not None and len(injection_specs) < _safe_int(dollar_values[1], 0):
                        duration = _parse_float(fields[1]) if len(fields) > 1 else None
                        if len(dollar_values) > 9:
                            injection_specs.append((first, duration))
                continue
            if line.startswith("#"):
                hash_values.append(_parse_float(line[1:]))
                continue
            if line.startswith("@"):
                fields = [field.strip() for field in line[1:].split(",")]
                current_block = int(fields[0])
                volume = _parse_float(fields[1]) if len(fields) > 1 else None
                duration = _parse_float(fields[2]) if len(fields) > 2 else None
                if volume is None and current_block > 0 and current_block - 1 < len(injection_specs):
                    volume = injection_specs[current_block - 1][0]
                if duration is None and current_block > 0 and current_block - 1 < len(injection_specs):
                    duration = injection_specs[current_block - 1][1]
                blocks[current_block] = (float(volume or 0.0), duration, [])
                continue
            if current_block is None:
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 3:
                continue
            try:
                blocks[current_block][2].append((float(fields[0]), float(fields[1]), float(fields[2])))
            except ValueError:
                continue

    metadata = {
        "number_of_injections": float(_safe_int(dollar_values[1], len(injection_specs)) if len(dollar_values) > 1 else len(injection_specs)),
        "target_temperature_c": float(_parse_float(dollar_values[3]) or 25.0) if len(dollar_values) > 3 else 25.0,
        "syringe_concentration_mM": float(hash_values[1] or 0.0) if len(hash_values) > 1 else 0.0,
        "cell_concentration_mM": float(hash_values[2] or 0.0) if len(hash_values) > 2 else 0.0,
        "cell_volume_mL": float(hash_values[3] or 0.0) if len(hash_values) > 3 else 0.0,
    }
    raw_blocks = tuple(
        RawInjectionBlock(
            number=number,
            volume_liter=volume_uL * LITERS_PER_MICROLITER,
            duration_s=duration_s,
            points=np.asarray(points, dtype=float),
        )
        for number, (volume_uL, duration_s, points) in sorted(blocks.items())
        if number > 0 and points
    )
    return metadata, raw_blocks


def _safe_int(value: str, default: int) -> int:
    try:
        return int(float(value.strip()))
    except (ValueError, AttributeError):
        return default


def integrate_raw_blocks(
    blocks: Iterable[RawInjectionBlock],
    *,
    tail_fraction: float = 0.20,
) -> np.ndarray:
    """Integrate raw injection blocks after subtracting a tail-median baseline."""
    heats: list[float] = []
    for block in blocks:
        points = block.points
        if points.ndim != 2 or points.shape[0] < 3:
            heats.append(np.nan)
            continue
        time = points[:, 0]
        power = points[:, 1]
        tail_count = max(3, int(round(tail_fraction * power.size)))
        baseline = float(np.median(power[-tail_count:]))
        # scipy's trapezoid works across NumPy versions (np.trapz was removed in NumPy 2.0).
        heats.append(float(trapezoid(power - baseline, time)))
    return np.asarray(heats, dtype=float)


def load_itc(
    path: str | Path,
    *,
    integrated_heats: str | Path | None = None,
    tail_fraction: float = 0.20,
    cell_concentration_mM: float | None = None,
    syringe_concentration_mM: float | None = None,
    cell_volume_mL: float | None = None,
    temperature_k: float | None = None,
    name: str | None = None,
) -> ITCExperiment:
    """Load a MicroCal `.itc` file, optionally using a paired integrated heat file."""
    path = _as_path(path)
    metadata, blocks = parse_itc_raw(path)

    if integrated_heats is not None:
        paired = load_dat(
            integrated_heats,
            cell_concentration_mM=cell_concentration_mM or metadata["cell_concentration_mM"],
            syringe_concentration_mM=syringe_concentration_mM or metadata["syringe_concentration_mM"],
            cell_volume_mL=cell_volume_mL or metadata["cell_volume_mL"],
            temperature_k=temperature_k or metadata["target_temperature_c"] + 273.15,
            name=name or path.stem,
        )
        return ITCExperiment(
            name=paired.name,
            injection_volumes_liter=paired.injection_volumes_liter,
            heats_microcalorie=paired.heats_microcalorie,
            cell_concentration_molar=paired.cell_concentration_molar,
            syringe_concentration_molar=paired.syringe_concentration_molar,
            cell_volume_liter=paired.cell_volume_liter,
            temperature_k=paired.temperature_k,
            source_path=path,
            raw_blocks=blocks,
        )

    heats = integrate_raw_blocks(blocks, tail_fraction=tail_fraction)
    injection_volumes = np.asarray([block.volume_liter for block in blocks], dtype=float)
    return ITCExperiment(
        name=name or path.stem,
        injection_volumes_liter=injection_volumes,
        heats_microcalorie=heats,
        cell_concentration_molar=float(cell_concentration_mM or metadata["cell_concentration_mM"]) * MOLAR_PER_MILLIMOLAR,
        syringe_concentration_molar=float(syringe_concentration_mM or metadata["syringe_concentration_mM"]) * MOLAR_PER_MILLIMOLAR,
        cell_volume_liter=float(cell_volume_mL or metadata["cell_volume_mL"]) * 1e-3,
        temperature_k=float(temperature_k or metadata["target_temperature_c"] + 273.15),
        source_path=path,
        raw_blocks=blocks,
    )


def load_experiment(path: str | Path, **kwargs) -> ITCExperiment:
    """Load `.DAT` or `.itc` by extension."""
    path = _as_path(path)
    suffix = path.suffix.lower()
    if suffix == ".dat":
        return load_dat(path, **kwargs)
    if suffix == ".itc":
        return load_itc(path, **kwargs)
    raise ValueError(f"Unsupported ITC data extension: {path.suffix}")


@dataclass(frozen=True)
class CuratedITCCurve:
    """A curated, injection-level ITC curve with pre-propagated cell concentrations.

    This is the data model for *sequential / multi-segment* titrations whose concentration bookkeeping
    cannot be expressed by the single-cell, single-syringe scan in the ``expected_heats`` models -- e.g.
    the TrpR L39E phosphate series, where the cell contents (and a buffer->ligand syringe switch) are
    carried across several raw ``.itc`` files. The curation step (see
    ``data/TrpR/curated_bayesian_regression``) propagates the dilution recursion

        f_i = 1 - v_i / V_cell;  D_i = f_i * D_{i-1};  L_i = f_i * L_{i-1} + (1 - f_i) * L_syringe

    once, storing the total concentrations **before** and **after** each injection. A likelihood then
    predicts the integrated heat of injection ``i`` directly from a model's ``enthalpy_density`` as

        q_i = V_cell * (H(after_i) - f_i * H(before_i)) * 1e9   [microcalories]   (+ nuisances),

    with no probabilistic scan. All concentrations are molar, volumes liters, heats microcalories.

    ``dimer_*`` are the curated TrpR **dimer** totals; coupled monomer-dimer models use
    ``2 * dimer_*`` as monomer-equivalent conserved protein (handled in the model builder).
    """

    name: str
    injection_volumes_liter: np.ndarray
    cell_volume_liter: float
    temperature_k: float
    syringe_ligand_molar: np.ndarray
    dimer_before_molar: np.ndarray
    ligand_before_molar: np.ndarray
    dimer_after_molar: np.ndarray
    ligand_after_molar: np.ndarray
    observed_heats_microcalorie: np.ndarray
    include_mask: np.ndarray
    source_files: tuple[str, ...] = ()
    source_path: Path | None = None

    @property
    def dilution_factors(self) -> np.ndarray:
        """Per-injection displacement factor ``f_i = 1 - v_i / V_cell``."""
        return 1.0 - self.injection_volumes_liter / self.cell_volume_liter

    @property
    def n_injections(self) -> int:
        return int(self.observed_heats_microcalorie.size)

    @property
    def n_included(self) -> int:
        return int(np.count_nonzero(self.include_mask))


def load_curated_itc_curve(path: str | Path, *, name: str | None = None) -> CuratedITCCurve:
    """Load a curated injection-summary ``.itc`` CSV (``# comment`` header + one row per injection).

    Expected columns (see ``data/TrpR/curated_bayesian_regression/README``): ``cell_volume_ml``,
    ``injection_volume_ul``, ``syringe_ligand_um``, ``dimer_um_before/after``, ``ligand_um_before/after``,
    ``observed_heat`` (microcalories), ``include_in_fit`` (1/0), ``temperature_mean_c``, ``source_file``.
    Units are converted to molar / liters; temperature is taken as the curve mean (these runs are held
    at ~25 C).
    """
    path = _as_path(path)
    header: list[str] | None = None
    rows: list[list[str]] = []
    with path.open(errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = [field.strip() for field in stripped.split(",")]
            if header is None:
                header = fields
                continue
            rows.append(fields)
    if header is None or not rows:
        raise ValueError(f"No curated injection rows found in {path}")

    index = {column: position for position, column in enumerate(header)}

    def column(name: str) -> np.ndarray:
        return np.asarray([row[index[name]] for row in rows], dtype=object)

    def floats(name: str) -> np.ndarray:
        return np.asarray([float(value) for value in column(name)], dtype=float)

    cell_volume_ml = floats("cell_volume_ml")
    temperature_c = floats("temperature_mean_c")
    source_files = tuple(dict.fromkeys(str(value) for value in column("source_file")))
    return CuratedITCCurve(
        name=name or path.stem,
        injection_volumes_liter=floats("injection_volume_ul") * LITERS_PER_MICROLITER,
        cell_volume_liter=float(np.median(cell_volume_ml)) * LITERS_PER_MILLILITER,
        temperature_k=float(np.mean(temperature_c)) + 273.15,
        syringe_ligand_molar=floats("syringe_ligand_um") * 1.0e-6,
        dimer_before_molar=floats("dimer_um_before") * 1.0e-6,
        ligand_before_molar=floats("ligand_um_before") * 1.0e-6,
        dimer_after_molar=floats("dimer_um_after") * 1.0e-6,
        ligand_after_molar=floats("ligand_um_after") * 1.0e-6,
        observed_heats_microcalorie=floats("observed_heat"),
        include_mask=floats("include_in_fit") > 0.5,
        source_files=source_files,
        source_path=path,
    )

