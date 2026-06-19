"""Physical constants and unit-conversion helpers."""

R_KCAL_PER_MOL_K = 0.00198720425864083
DEFAULT_TEMPERATURE_K = 298.15
MICROCALORIES_PER_KCAL = 1.0e9
LITERS_PER_MILLILITER = 1.0e-3
LITERS_PER_MICROLITER = 1.0e-6
MOLAR_PER_MILLIMOLAR = 1.0e-3


def beta_mol_per_kcal(temperature_k: float = DEFAULT_TEMPERATURE_K) -> float:
    """Return inverse thermal energy in mol/kcal."""
    return 1.0 / (R_KCAL_PER_MOL_K * temperature_k)

