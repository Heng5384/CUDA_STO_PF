#!/usr/bin/env python3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "params" / "pf_only_x_q_validation"
OUT.mkdir(parents=True, exist_ok=True)


def replace_or_append(text: str, key: str, value: str) -> str:
    lines = text.splitlines()
    prefix = key + "="
    for i, line in enumerate(lines):
        if line.startswith(prefix):
            lines[i] = prefix + value
            break
    else:
        lines.append(prefix + value)
    return "\n".join(lines) + "\n"


dts = [("0p002", "0.002", 500), ("0p001", "0.001", 960),
       ("0p0005", "0.0005", 1880)]
rows = []
for dt_tag, dt, nsteps in dts:
    base = (ROOT / "params" / "pf_only_baseline_closure" / "remediation" /
            f"T400_PFBASE_storage_weighted_dt{dt_tag}.params")
    source = base.read_text()
    for ds in (5, 20):
        tag = f"T400_PFX_Ds{ds}_eps0p10_dt{dt_tag}"
        text = replace_or_append(source, "pf_y_update_mode", "x_transport_projection_split")
        text = replace_or_append(text, "pf_matrix_storage_floor", "0.10")
        text = replace_or_append(text, "pf_composition_stabilizer_Dalpha_multiplier", str(ds))
        path = OUT / f"{tag}.params"
        path.write_text(text)
        rows.append((tag, path, dt, nsteps, "Ds_sensitivity"))

base = (ROOT / "params" / "pf_only_baseline_closure" / "remediation" /
        "T400_PFBASE_storage_weighted_dt0p0005.params")
source = base.read_text()
for eps_tag, eps in (("0p05", "0.05"), ("0p20", "0.20")):
    tag = f"T400_PFX_Ds10_eps{eps_tag}_dt0p0005"
    text = replace_or_append(source, "pf_y_update_mode", "x_transport_projection_split")
    text = replace_or_append(text, "pf_matrix_storage_floor", eps)
    text = replace_or_append(text, "pf_composition_stabilizer_Dalpha_multiplier", "10")
    path = OUT / f"{tag}.params"
    path.write_text(text)
    rows.append((tag, path, "0.0005", 1880, "epsilon_sensitivity"))

manifest = OUT / "case_manifest.csv"
manifest.write_text(
    "case,param_file,dt,nsteps,purpose\n" +
    "".join(f"{tag},{path.relative_to(ROOT)},{dt},{nsteps},{purpose}\n"
            for tag, path, dt, nsteps, purpose in rows)
)
print(manifest)
