#!/usr/bin/env python3
"""Host-only checks for CUDA build-provenance generation.

This test deliberately does not compile CUDA.  It detects malformed or
unbound contract/fixture metadata before a cluster build is submitted; a
failure means the build scripts must be corrected rather than launching a
runtime job.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools" / "generate_pf_cuda_build_provenance.py"
FINALIZER = ROOT / "tools" / "finalize_pf_cuda_build_provenance.py"


class CudaBuildProvenanceTest(unittest.TestCase):
    def test_generator_embeds_contract_and_validation_schemas(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            header = temp / "pf_cuda_build_provenance_v1.h"
            manifest = temp / "main_cuda.provenance.json"
            command = [
                sys.executable,
                str(GENERATOR),
                "--source-root",
                str(ROOT),
                "--contract",
                str(ROOT / "contracts" / "pf_kwn_validation_contract_v1.json"),
                "--contract-header",
                str(ROOT / "generated" / "pf_kwn_validation_contract_v1.h"),
                "--fixture-spec",
                str(
                    ROOT
                    / "data"
                    / "qualification"
                    / "pf_mass_conserving_library_handoff_v1"
                    / "six_particle_96cube_spec.json"
                ),
                "--output-header",
                str(header),
                "--output-json",
                str(manifest),
                "--nvcc",
                "definitely-not-an-nvcc-binary",
                "--host-cxx",
                "c++",
                "--cuda-root",
                "/not-a-cuda-install",
                "--cuda-arch",
                "sm_120",
                "--nvccflags=-arch=sm_120 -O3",
                "--includes=-I/not-a-cuda-install/include",
                "--ldflags=-L/not-a-cuda-install/lib64",
                "--ldlibs=-lcufft -lcudart -lm",
            ]
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            document = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(document["schema_version"], "PF_CUDA_BUILD_PROVENANCE_V1")
            self.assertFalse(document["controlled_binary_eligible"])
            self.assertEqual(
                document["contract"]["canonical_hash"],
                "d0ff02973ab0f737043e1a40d4f69893a469cbfe2bc4cd22f9e6a410bd0b1333",
            )
            self.assertEqual(
                document["fixture"]["schema"],
                "PF_MASS_CONSERVING_LIBRARY_HANDOFF_SPEC_V1",
            )
            self.assertEqual(document["fixture"]["target_grid"], [96, 96, 96])
            self.assertEqual(document["checkpoint"]["schema"], "PF_ZERO_MODE_CHECKPOINT_V6")
            self.assertEqual(
                document["checkpoint"]["auxiliary_sidecar_schema"],
                "PF_AUXILIARY_HANDOFF_V2_SIDECAR_V1",
            )
            header_text = header.read_text(encoding="utf-8")
            self.assertIn("PF_CUDA_BUILD_PROVENANCE_JSON", header_text)
            self.assertIn(document["contract"]["canonical_hash"], header_text)

            compiler = shutil.which("c++")
            if compiler is None:
                self.skipTest("host C++ compiler is unavailable")
            probe_source = temp / "provenance_probe.cpp"
            probe_binary = temp / "provenance_probe"
            probe_source.write_text(
                "#include <cstdio>\n"
                "#include \"pf_cuda_build_provenance_v1.h\"\n"
                "int main() { std::puts(PF_CUDA_BUILD_PROVENANCE_JSON); }\n",
                encoding="utf-8",
            )
            subprocess.run(
                [compiler, "-std=c++14", "-I", str(temp), str(probe_source), "-o", str(probe_binary)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            compiled_json = subprocess.run(
                [str(probe_binary)], check=True, capture_output=True, text=True
            ).stdout
            self.assertEqual(json.loads(compiled_json), document)

            # The finalizer is intentionally independent from CUDA: it must
            # hash an already-compiled artifact and verify the exact JSON
            # embedded in the generated header.  A tiny temporary byte stream
            # exercises that binding without claiming it is a CUDA binary.
            binary = temp / "main_cuda"
            binary.write_bytes(b"host-only provenance fixture\n")
            finalized = temp / "controlled_binary_manifest.json"
            subprocess.run(
                [
                    sys.executable,
                    str(FINALIZER),
                    "--binary",
                    str(binary),
                    "--build-provenance",
                    str(manifest),
                    "--build-header",
                    str(header),
                    "--output",
                    str(finalized),
                    "--allow-uncontrolled",
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            finalized_document = json.loads(finalized.read_text(encoding="utf-8"))
            self.assertEqual(
                finalized_document["status"], "UNCONTROLLED_CUDA_BINARY_PROVENANCE_V1"
            )
            listed_paths = {
                row["path"] for row in finalized_document["artifacts"]["build_directory"]["files"]
            }
            self.assertTrue(
                {"main_cuda", "main_cuda.provenance.json", "pf_cuda_build_provenance_v1.h"}
                <= listed_paths
            )

    def test_clean_source_can_produce_a_controlled_binary_manifest(self) -> None:
        """Exercise the clean-source branch without needing NVCC or a GPU."""

        if shutil.which("git") is None:
            self.skipTest("git is unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "source"
            artifacts = temp / "artifacts"
            (source / "tools").mkdir(parents=True)
            (source / "contracts").mkdir()
            fixture_dir = (
                source
                / "data"
                / "qualification"
                / "pf_mass_conserving_library_handoff_v1"
            )
            fixture_dir.mkdir(parents=True)
            shutil.copy2(
                ROOT / "tools" / "generate_pf_contract_header.py",
                source / "tools" / "generate_pf_contract_header.py",
            )
            shutil.copy2(
                ROOT / "contracts" / "pf_kwn_validation_contract_v1.json",
                source / "contracts" / "pf_kwn_validation_contract_v1.json",
            )
            shutil.copy2(
                ROOT
                / "data"
                / "qualification"
                / "pf_mass_conserving_library_handoff_v1"
                / "six_particle_96cube_spec.json",
                fixture_dir / "six_particle_96cube_spec.json",
            )
            (source / ".gitignore").write_text("generated/\n", encoding="utf-8")
            for command in (
                ["git", "init"],
                ["git", "config", "user.email", "provenance-test@example.invalid"],
                ["git", "config", "user.name", "CUDA provenance test"],
                ["git", "add", "."],
                ["git", "commit", "-m", "test fixture"],
            ):
                subprocess.run(command, cwd=source, check=True, capture_output=True, text=True)
            contract_header = source / "generated" / "pf_kwn_validation_contract_v1.h"
            subprocess.run(
                [
                    sys.executable,
                    str(source / "tools" / "generate_pf_contract_header.py"),
                    "--contract",
                    str(source / "contracts" / "pf_kwn_validation_contract_v1.json"),
                    "--header",
                    str(contract_header),
                ],
                cwd=source,
                check=True,
                capture_output=True,
                text=True,
            )
            artifacts.mkdir()
            header = artifacts / "pf_cuda_build_provenance_v1.h"
            manifest = artifacts / "main_cuda.provenance.json"
            subprocess.run(
                [
                    sys.executable,
                    str(GENERATOR),
                    "--source-root",
                    str(source),
                    "--contract",
                    str(source / "contracts" / "pf_kwn_validation_contract_v1.json"),
                    "--contract-header",
                    str(contract_header),
                    "--fixture-spec",
                    str(fixture_dir / "six_particle_96cube_spec.json"),
                    "--output-header",
                    str(header),
                    "--output-json",
                    str(manifest),
                    "--nvcc",
                    "definitely-not-an-nvcc-binary",
                    "--host-cxx",
                    "c++",
                    "--cuda-root",
                    "/not-a-cuda-install",
                    "--cuda-arch",
                    "sm_120",
                    "--nvccflags=-arch=sm_120 -O3",
                    "--includes=-I/not-a-cuda-install/include",
                    "--ldflags=-L/not-a-cuda-install/lib64",
                    "--ldlibs=-lcufft -lcudart -lm",
                    "--require-clean",
                ],
                cwd=source,
                check=True,
                capture_output=True,
                text=True,
            )
            document = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertTrue(document["source"]["clean"])
            self.assertTrue(document["controlled_binary_eligible"])
            binary = artifacts / "main_cuda"
            binary.write_bytes(b"host-only controlled provenance fixture\n")
            finalized = artifacts / "controlled_binary_manifest.json"
            subprocess.run(
                [
                    sys.executable,
                    str(FINALIZER),
                    "--binary",
                    str(binary),
                    "--build-provenance",
                    str(manifest),
                    "--build-header",
                    str(header),
                    "--output",
                    str(finalized),
                ],
                cwd=source,
                check=True,
                capture_output=True,
                text=True,
            )
            finalized_document = json.loads(finalized.read_text(encoding="utf-8"))
            self.assertEqual(
                finalized_document["status"], "PASS_CONTROLLED_CUDA_BINARY_PROVENANCE_V1"
            )


if __name__ == "__main__":
    unittest.main()
