#!/usr/bin/env python3
"""Write compact provenance for the frozen PF--KWN validation contract.

The canonical JSON remains source-controlled in ``contracts/``.  This helper
emits only a reproducible result-side manifest, avoiding a second editable
copy while making the artifact directory self-describing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kwn_mvp.contract import load_validation_contract  # noqa: E402


def main() -> int:
    contract_path = ROOT / "contracts" / "pf_kwn_validation_contract_v1.json"
    header_check = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "generate_pf_contract_header.py"), "--check"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    generated = json.loads(header_check.stdout)
    contract = load_validation_contract(contract_path)
    if generated["hash"] != contract.sha256:
        raise RuntimeError("generated PF header hash differs from the KWN contract hash")
    output = ROOT / "outputs" / "kwn_pf_state_closure_v1" / "contracts"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "PF_KWN_VALIDATION_CONTRACT_ARTIFACT_V1",
        "contract_schema_version": contract.data["schema_version"],
        "contract_path": str(contract_path),
        "contract_hash": contract.sha256,
        "generated_header": str(ROOT / "generated" / "pf_kwn_validation_contract_v1.h"),
        "generated_header_hash": generated["hash"],
        "scope": "VALIDATION_CONTROL_ONLY_NOT_HISTORICAL_AS_RUN",
    }
    path = output / "validation_contract_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
