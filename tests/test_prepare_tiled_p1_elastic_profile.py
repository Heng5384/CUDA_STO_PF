import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_tiled_p1_elastic_profile.py"
SPEC = importlib.util.spec_from_file_location("tiled_profile", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


class TiledP1ElasticProfileTests(unittest.TestCase):
    def test_ascii_vtk_read_and_periodic_tile(self):
        source = np.arange(8, dtype=np.float64).reshape((2, 2, 2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "field.vtk"
            path.write_text(
                "# vtk DataFile Version 3.0\n"
                "test\nASCII\nDATASET STRUCTURED_POINTS\n"
                "DIMENSIONS 2 2 2\nASPECT_RATIO 1 1 1\nORIGIN 0 0 0\n"
                "POINT_DATA 8\nSCALARS f double 1\nLOOKUP_TABLE default\n"
                + " ".join(str(value) for value in source.ravel())
                + "\n"
            )
            loaded, dimensions = module.read_ascii_structured_points(path)
        self.assertEqual(dimensions, (2, 2, 2))
        np.testing.assert_array_equal(loaded, source)
        tiled = module.tile_field(loaded, 4)
        self.assertEqual(tiled.shape, (4, 4, 4))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    self.assertEqual(tiled[i, j, k], source[i % 2, j % 2, k % 2])

    def test_rejects_nondivisible_target(self):
        with self.assertRaises(ValueError):
            module.tile_field(np.zeros((3, 3, 3)), 32)

    def test_metadata_closes_raw_init_contract(self):
        metadata = module.build_metadata(32, "accepted_source", (16, 16, 16))
        self.assertEqual(metadata["Nx"], 32)
        self.assertEqual(metadata["dx_nm"], 0.05)
        self.assertEqual(metadata["interface_width_nm"], 0.6)
        self.assertEqual(metadata["authoritative_state"], "Ctot")
        self.assertEqual(
            metadata["adjoint_identity_mode"], "D_equals_negative_G_star_exact"
        )
        self.assertFalse(metadata["physical_benchmark_evidence"])

    def test_ctot_reconstruction_uses_stable_near_beta_storage(self):
        phi = np.array([0.0, 0.5, 0.99999, 1.0])
        x_b = np.full_like(phi, 0.007830539083594734)
        ctot = module.reconstruct_ctot(phi, x_b)
        alpha = module.alpha_stable(phi)
        np.testing.assert_allclose(
            ctot, 1.0 - alpha * (1.0 - x_b), rtol=0, atol=2.0e-16
        )
        self.assertEqual(ctot[0], x_b[0])
        self.assertEqual(ctot[-1], 1.0)
        self.assertTrue(np.all(ctot >= 1.0 - alpha))


if __name__ == "__main__":
    unittest.main()
