"""Formal-launch provenance when the CPU compute image has no Git binary."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from scripts import run_kwn_characteristic_reference_v1 as driver


def _batch_identity(*, clean: str = "1") -> dict[str, str]:
    return {
        "KWN_LAUNCH_GIT_HEAD": "639a3b79f318d84641f72cac6d380eac1b8d48ec",
        "KWN_LAUNCH_GIT_BRANCH": driver.REQUIRED_BRANCH,
        "KWN_LAUNCH_GIT_CLEAN": clean,
        "KWN_LAUNCH_FROZEN_START_ANCESTOR": driver.FROZEN_START_COMMIT,
        "KWN_LAUNCH_SOURCE_ROOT": str(driver.ROOT),
    }


class CharacteristicLaunchProvenanceTests(unittest.TestCase):
    """A submit-host Git record may transport identity, never weaken it."""

    def test_no_compute_node_git_uses_checked_submit_identity(self) -> None:
        with patch.dict(os.environ, _batch_identity(), clear=False), patch.object(
            driver, "_git", side_effect=FileNotFoundError()
        ):
            identity = driver._launch_source_identity()
        self.assertEqual(identity["head"], _batch_identity()["KWN_LAUNCH_GIT_HEAD"])
        self.assertEqual(identity["branch"], driver.REQUIRED_BRANCH)
        self.assertTrue(identity["clean"])
        self.assertTrue(identity["frozen_start_is_ancestor"])
        self.assertEqual(
            identity["mode"], "SUBMISSION_GIT_PROVENANCE_NO_COMPUTE_NODE_GIT"
        )

    def test_no_compute_node_git_rejects_dirty_submit_identity(self) -> None:
        with patch.dict(os.environ, _batch_identity(clean="0"), clear=False), patch.object(
            driver, "_git", side_effect=FileNotFoundError()
        ):
            with self.assertRaisesRegex(driver.WorkflowError, "clean source tree"):
                driver._launch_source_identity()


if __name__ == "__main__":
    unittest.main()
