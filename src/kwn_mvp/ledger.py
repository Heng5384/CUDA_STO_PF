"""Independent four-bucket inventory ledger for KWN and PF handoff."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from .population_metrics import PopulationMetricError, close_matrix_from_precipitates
from .populations import Population


class InventoryError(RuntimeError):
    """Raised when an update cannot close the pseudo-binary B inventory."""


@dataclass(frozen=True)
class InventorySnapshot:
    """All B inventories on an SI mol m^-3 basis."""

    total_mol_m3: float
    matrix_mol_m3: float
    gp_mol_m3: float
    beta_subgrid_mol_m3: float
    beta_resolved_mol_m3: float
    residual_mol_m3: float
    relative_residual: float
    matrix_fraction: float


class InventoryLedger:
    """Exact algebraic ledger with no clamp-based mass correction.

    The matrix amount is ``(1-f_g-f_beta)*xB_alpha/Vm_alpha``.  Each
    population uses its own molar volume and composition, so GP and beta are
    never silently treated as identical material.
    """

    def __init__(
        self,
        *,
        matrix_molar_volume_m3_mol: float,
        total_b_mol_m3: float,
        tolerance_relative: float,
    ) -> None:
        if matrix_molar_volume_m3_mol <= 0.0:
            raise InventoryError("matrix molar volume must be positive")
        if total_b_mol_m3 < 0.0:
            raise InventoryError("total inventory cannot be negative")
        if tolerance_relative <= 0.0:
            raise InventoryError("relative inventory tolerance must be positive")
        self.matrix_molar_volume_m3_mol = float(matrix_molar_volume_m3_mol)
        self.total_b_mol_m3 = float(total_b_mol_m3)
        self.tolerance_relative = float(tolerance_relative)

    @staticmethod
    def _population_by_name(populations: Iterable[Population], name: str) -> Population:
        """Return a named population or fail with an actionable message."""

        for population in populations:
            if population.parameters.name == name:
                return population
        raise InventoryError(f"Missing required population '{name}'")

    def recover_matrix_xb(self, populations: Iterable[Population]) -> float:
        """Recover matrix xB exactly from fixed total inventory.

        This is the ledger operation after every finite-volume/source update.
        It does not clamp an out-of-range composition; such an event is an
        invalid physical/configuration state and raises ``InventoryError``.
        """

        population_list = list(populations)
        gp = self._population_by_name(population_list, "g")
        beta = self._population_by_name(population_list, "beta")
        fraction = 1.0 - gp.volume_fraction() - beta.volume_fraction()
        precipitate_b = gp.b_inventory_mol_m3() + beta.b_inventory_mol_m3()
        try:
            closure = close_matrix_from_precipitates(
                total_b_mol_m3=self.total_b_mol_m3,
                matrix_molar_volume_m3_mol=self.matrix_molar_volume_m3_mol,
                precipitate_volume_fraction=1.0 - fraction,
                precipitate_inventory_mol_m3=precipitate_b,
            )
        except PopulationMetricError as error:
            raise InventoryError(str(error)) from error
        return closure.matrix_xb

    def snapshot(
        self,
        *,
        matrix_xb: float,
        populations: Iterable[Population],
        beta_resolved_fraction: float | None = None,
    ) -> InventorySnapshot:
        """Return a closed inventory snapshot, optionally splitting beta at handoff."""

        population_list = list(populations)
        gp = self._population_by_name(population_list, "g")
        beta = self._population_by_name(population_list, "beta")
        matrix_fraction = 1.0 - gp.volume_fraction() - beta.volume_fraction()
        if matrix_fraction < 0.0:
            raise InventoryError("Population volume fraction exceeds total material volume")
        gp_inventory = gp.b_inventory_mol_m3()
        beta_total = beta.b_inventory_mol_m3()
        try:
            # The shared closure validates the physical state and defines the
            # inverse used by both Eulerian and cohort paths.  Snapshot still
            # evaluates the supplied matrix composition to expose any drift.
            close_matrix_from_precipitates(
                total_b_mol_m3=self.total_b_mol_m3,
                matrix_molar_volume_m3_mol=self.matrix_molar_volume_m3_mol,
                precipitate_volume_fraction=1.0 - matrix_fraction,
                precipitate_inventory_mol_m3=gp_inventory + beta_total,
            )
        except PopulationMetricError as error:
            raise InventoryError(str(error)) from error
        matrix = matrix_fraction * matrix_xb / self.matrix_molar_volume_m3_mol
        fraction = 0.0 if beta_resolved_fraction is None else float(beta_resolved_fraction)
        if not 0.0 <= fraction <= 1.0:
            raise InventoryError("beta_resolved_fraction must lie in [0, 1]")
        beta_resolved = beta_total * fraction
        beta_subgrid = beta_total - beta_resolved
        reconstructed = matrix + gp_inventory + beta_subgrid + beta_resolved
        residual = reconstructed - self.total_b_mol_m3
        denominator = max(abs(self.total_b_mol_m3), 1.0e-300)
        snapshot = InventorySnapshot(
            total_mol_m3=self.total_b_mol_m3,
            matrix_mol_m3=matrix,
            gp_mol_m3=gp_inventory,
            beta_subgrid_mol_m3=beta_subgrid,
            beta_resolved_mol_m3=beta_resolved,
            residual_mol_m3=residual,
            relative_residual=abs(residual) / denominator,
            matrix_fraction=matrix_fraction,
        )
        if snapshot.relative_residual > self.tolerance_relative:
            raise InventoryError(
                f"Inventory residual {snapshot.relative_residual:.3e} exceeds "
                f"tolerance {self.tolerance_relative:.3e}"
            )
        return snapshot

    def component_dict(self, *, matrix_xb: float, populations: Iterable[Population]) -> Dict[str, float]:
        """Return a machine-friendly full ledger without a beta handoff split."""

        item = self.snapshot(matrix_xb=matrix_xb, populations=populations)
        return {
            "C_B_total_mol_m3": item.total_mol_m3,
            "C_B_matrix_mol_m3": item.matrix_mol_m3,
            "C_B_GP_mol_m3": item.gp_mol_m3,
            "C_B_beta_subgrid_mol_m3": item.beta_subgrid_mol_m3,
            "C_B_beta_resolved_mol_m3": item.beta_resolved_mol_m3,
            "residual_mol_m3": item.residual_mol_m3,
            "relative_residual": item.relative_residual,
        }
