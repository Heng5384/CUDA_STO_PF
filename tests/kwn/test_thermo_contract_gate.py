"""Fail-closed regression tests for PF-linked KWN thermodynamic requests."""

from __future__ import annotations

import unittest

from kwn_mvp.thermo_adapter import (
    PFContractReference,
    ThermodynamicContractBlocked,
    build_equilibrium_adapter,
)


class ThermodynamicContractGateTest(unittest.TestCase):
    """Ensure the MVP never silently selects legacy or exact-candidate PF data."""

    def test_unresolved_pf_contract_cannot_construct_an_adapter(self) -> None:
        """A P0 contract conflict blocks PF-linked KWN before numerical execution."""

        reference = PFContractReference(
            status="BLOCKED_CONTRACT_CONFLICT",
            source="thermodynamic_contract_freeze_v1/07_frozen_thermodynamic_contract.yaml",
            source_commit="1c08f9ee011b31e0cd4d82749e58a8f69ebd2204",
        )
        with self.assertRaises(ThermodynamicContractBlocked):
            build_equilibrium_adapter(
                mode="pf_contract",
                temperature_k=653.15,
                pf_contract=reference,
            )
