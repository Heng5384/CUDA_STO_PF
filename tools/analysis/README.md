Analysis and post-processing scripts live here.

Run them from the repository root so their existing relative-path behavior
remains unchanged. Examples:

`python3 tools/analysis/compute_schur_rc_predictions.py`

`python3 tools/analysis/analyze_cnt_peak_table.py`

`python3 tools/analysis/organize_constraint_results_by_strain.py`

Standardized CNT workflow entry points:

- `python3 tools/analysis/setup_cnt_workflow.py ...`
- `python3 tools/analysis/summarize_cnt_scan_from_guide.py ...`
- `python3 tools/analysis/prepare_continue_dynamic_guide.py ...`
- `python3 tools/analysis/summarize_continue_from_guide.py ...`

Workflow documentation:

- [`critical_radius_continue_workflow.md`](/Users/heng/Documents/GitHub/CUDA_STO_PF/tools/analysis/critical_radius_continue_workflow.md)
