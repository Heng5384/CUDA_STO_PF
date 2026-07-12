#!/usr/bin/env python3
import csv
import json
import math
import pathlib
import re
import statistics
import subprocess
import sys


ROOT = pathlib.Path("Results/codex_external_prediction_validation")
BASE_INPUT_JSON = pathlib.Path("physical_inputs.example.json")
SEEDS = [101, 102, 103, 104, 105]
NSTEPS = 30
DT = 0.01


def run_matrix():
    ROOT.mkdir(parents=True, exist_ok=True)
    param_dir = ROOT / "pf_params_by_temperature"
    param_dir.mkdir(parents=True, exist_ok=True)
    params_by_temp = {}
    base_inputs = json.loads(BASE_INPUT_JSON.read_text(encoding="utf-8"))
    for temp_k in [350, 380, 420]:
        temp_c = temp_k - 273.15
        inputs = dict(base_inputs)
        inputs["temperature_C"] = temp_c
        input_json = param_dir / f"physical_inputs_T{temp_k}K.json"
        output_params = param_dir / f"pf_input_T{temp_k}K.params"
        input_json.write_text(json.dumps(inputs, indent=2), encoding="utf-8")
        subprocess.run([
            "python3", "Unit_Psedobinary.py",
            "--input-json", str(input_json),
            "--output-pf-param-file", str(output_params),
            "--no-summary",
        ], check=True)
        params_by_temp[temp_k] = str(output_params)

    conditions = []
    for temp_k in [350, 380, 420]:
        conditions.append({"axis": "temperature", "T": temp_k, "xB": 0.03, "strain": 0.0, "elastic": 0})
    for xb in [0.02, 0.04]:
        conditions.append({"axis": "composition", "T": 380, "xB": xb, "strain": 0.0, "elastic": 0})
    for strain in [0.0, 0.005, 0.01]:
        conditions.append({"axis": "strain", "T": 380, "xB": 0.03, "strain": strain, "elastic": 1})
    conditions = sorted(conditions, key=lambda c: (c["axis"], c["T"], c["xB"], c["strain"]))

    manifest = []
    for cond in conditions:
        temp_c = cond["T"] - 273.15
        x_tag = str(cond["xB"]).replace(".", "p")
        s_tag = str(cond["strain"]).replace(".", "p")
        tag_base = f"ext_{cond['axis']}_T{cond['T']}_xB{x_tag}_exx{s_tag}"
        for seed in SEEDS:
            tag = f"{tag_base}_seed{seed}"
            log = ROOT / f"{tag}.log"
            cmd = [
                "./main_cuda", "32", "32", "32", f"{DT}", f"{NSTEPS}", "30", "30", str(cond["elastic"]),
                "--pf-param-file", params_by_temp[cond["T"]],
                "--init-case-tag", tag,
                "--enable-gp-assisted-beta-nucleation",
                "--enable-gp-stochastic-selection",
                "--gp-site-mode", "grid",
                "--gp-n-sites", "50",
                "--gp-stochastic-k0", "10",
                "--gp-stochastic-S-GP", "0.01",
                "--gp-stochastic-deltaG-homo-kBT", "198",
                "--gp-seed", str(seed),
                "--gp-site-B-mass-equiv", "0.8",
                "--gp-release-radius-nm", "1.0",
                "--gp-debug-beta-seed-radius", "0.05",
                "--gp-debug-beta-seed-iface-width", "0.03",
                "--gp-initial-mass-mode", "total_composition_fixed",
                "--gp-release-mode", "release_to_beta_first",
                "--gp-release-kernel", "compact_spherical",
                "--gp-debug-xB-max", "0.08",
                "--ic-23d-xB-out", f"{cond['xB']:.12g}",
                "--external-strain-exx", f"{cond['strain']:.12g}",
            ]
            with log.open("w") as fp:
                rc = subprocess.run(cmd, stdout=fp, stderr=subprocess.STDOUT).returncode
            row = dict(cond)
            row.update({"seed": seed, "tag": tag, "log": str(log), "returncode": rc})
            manifest.append(row)
            print(f"{tag} rc={rc}")
            if rc != 0:
                raise SystemExit(rc)

    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def case_dir_for(item):
    log_text = pathlib.Path(item["log"]).read_text(errors="replace")
    m = re.search(r"case_output_dir\s*:\s*(\S+)", log_text)
    if m:
        path = pathlib.Path(m.group(1))
        if path.exists():
            return path
    matches = list(pathlib.Path("Results").glob(f"**/{item['tag']}"))
    if not matches:
        raise FileNotFoundError(item["tag"])
    return matches[0]


def entropy_from_separation(path):
    with path.open() as fp:
        probs = [float(r["P_i_new"]) for r in csv.DictReader(fp) if float(r["P_i_new"]) > 0.0]
    return -sum(p * math.log(p) for p in probs)


def top1_from_separation(path):
    with path.open() as fp:
        rows = list(csv.DictReader(fp))
    top = max(rows, key=lambda r: float(r["P_i_new"]))
    return int(float(top["site_id"]))


def analyze(manifest):
    grouped = {}
    for item in manifest:
        key = (item["T"], item["xB"], item["strain"])
        grouped.setdefault(key, []).append(item)

    out_rows = []
    for (temp_k, xb, strain), items in sorted(grouped.items()):
        first_times = []
        event_count = 0
        gp_consumed = 0.0
        entropies = []
        top_ids = []
        mass_drifts = []
        for item in items:
            cdir = case_dir_for(item)
            event_log = cdir / "gp_stochastic_event_log.csv"
            assisted_log = cdir / "gp_assisted_event_log.csv"
            sep_csv = cdir / "gp_strong_physics_separation.csv"
            scaling_csv = cdir / "gp_multi_site_scaling_analysis.csv"
            with event_log.open() as fp:
                events = list(csv.DictReader(fp))
            accepted = [r for r in events if r["event_status"] == "accepted"]
            event_count += len(accepted)
            if accepted:
                first_times.append(float(accepted[0]["time"]))
            with assisted_log.open() as fp:
                assisted_events = list(csv.DictReader(fp))
            gp_consumed += sum(float(r["mass_from_gp"]) for r in assisted_events if r["status"] == "accepted")
            entropies.append(entropy_from_separation(sep_csv))
            top_ids.append(top1_from_separation(sep_csv))
            with scaling_csv.open() as fp:
                rows = list(csv.DictReader(fp))
            if rows:
                mass_drifts.append(abs(float(rows[-1]["mass_drift_max"])))
            text = pathlib.Path(item["log"]).read_text(errors="replace")
            m = re.search(r"mass_conservation_under_N_sites_verified=true max_abs_rel_drift=([0-9.eE+-]+)", text)
            if m:
                mass_drifts.append(abs(float(m.group(1))))

        total_time = len(items) * NSTEPS * DT
        j_beta = event_count / total_time if total_time > 0 else math.nan
        gp_rate = gp_consumed / total_time if total_time > 0 else math.nan
        top1 = statistics.mode(top_ids)
        out_rows.append({
            "T": temp_k,
            "xB": xb,
            "strain": strain,
            "J_beta": j_beta,
            "t_first_event_mean": statistics.mean(first_times) if first_times else math.nan,
            "t_first_event_std": statistics.pstdev(first_times) if len(first_times) > 1 else 0.0 if first_times else math.nan,
            "entropy": statistics.mean(entropies),
            "top1_site_id": top1,
            "GP_consumed_rate": gp_rate,
            "mass_drift": max(mass_drifts) if mass_drifts else math.nan,
            "n_runs": len(items),
            "n_events": event_count,
            "top1_unique_count": len(set(top_ids)),
        })

    csv_path = ROOT / "external_prediction_validation.csv"
    fieldnames = [
        "T", "xB", "strain", "J_beta", "t_first_event_mean", "t_first_event_std",
        "entropy", "top1_site_id", "GP_consumed_rate", "mass_drift",
    ]
    with csv_path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow({k: row[k] for k in fieldnames})

    detail_path = ROOT / "external_prediction_validation_detail.json"
    detail_path.write_text(json.dumps(out_rows, indent=2, allow_nan=True), encoding="utf-8")
    return out_rows, csv_path


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    else:
        manifest = run_matrix()
    rows, csv_path = analyze(manifest)
    print(csv_path)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
