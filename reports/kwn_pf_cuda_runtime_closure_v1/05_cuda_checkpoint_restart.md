# 05 CUDA checkpoint/restart

Final restart comparisons are all required to be within `1e-14` for their recorded fields. The compact audit records:

```json
[
  {
    "auxiliary_populations": {
      "GP": {
        "bin_count": 0,
        "decoded_bins_exact": true,
        "inventory_exact": true,
        "inventory_mol": 0.0
      },
      "beta_subgrid": {
        "bin_count": 0,
        "decoded_bins_exact": true,
        "inventory_exact": true,
        "inventory_mol": 0.0
      }
    },
    "elastic_warm_state": {
      "byte_count": 10838016,
      "byte_identical": true,
      "present": true
    },
    "fields": {
      "Y": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      },
      "dY_dt_prev": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      },
      "phi": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      },
      "xB": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      }
    },
    "label": "A 6h continuous/restart",
    "left": "/data/home/luozhiheng/tmp/kwn_pf_cuda_runtime_closure_v1_097a69582ca3_20260901T080254Z/run_r4/restart_qualification/A/continuous_6h.pfzck",
    "right": "/data/home/luozhiheng/tmp/kwn_pf_cuda_runtime_closure_v1_097a69582ca3_20260901T080254Z/run_r4/restart_qualification/A/restart_6h.pfzck",
    "v6_header_and_provenance": {
      "all_fields_exact": true,
      "compared_field_count": 49,
      "mismatched_fields": []
    }
  },
  {
    "auxiliary_populations": {
      "GP": {
        "bin_count": 2,
        "decoded_bins_exact": true,
        "inventory_exact": true,
        "inventory_mol": 0.0
      },
      "beta_subgrid": {
        "bin_count": 2,
        "decoded_bins_exact": true,
        "inventory_exact": true,
        "inventory_mol": 0.0
      }
    },
    "elastic_warm_state": {
      "byte_count": 10838016,
      "byte_identical": true,
      "present": true
    },
    "fields": {
      "Y": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      },
      "dY_dt_prev": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      },
      "phi": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      },
      "xB": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      }
    },
    "label": "B 6h continuous/restart",
    "left": "/data/home/luozhiheng/tmp/kwn_pf_cuda_runtime_closure_v1_097a69582ca3_20260901T080254Z/run_r4/restart_qualification/B/continuous_6h.pfzck",
    "right": "/data/home/luozhiheng/tmp/kwn_pf_cuda_runtime_closure_v1_097a69582ca3_20260901T080254Z/run_r4/restart_qualification/B/restart_6h.pfzck",
    "v6_header_and_provenance": {
      "all_fields_exact": true,
      "compared_field_count": 49,
      "mismatched_fields": []
    }
  },
  {
    "auxiliary_populations": {
      "GP": {
        "bin_count": 2,
        "decoded_bins_exact": true,
        "inventory_exact": true,
        "inventory_mol": 1.309926818847619e-21
      },
      "beta_subgrid": {
        "bin_count": 2,
        "decoded_bins_exact": true,
        "inventory_exact": true,
        "inventory_mol": 0.0
      }
    },
    "elastic_warm_state": {
      "byte_count": 10838016,
      "byte_identical": true,
      "present": true
    },
    "fields": {
      "Y": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      },
      "dY_dt_prev": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      },
      "phi": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      },
      "xB": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      }
    },
    "label": "E 6h continuous/restart",
    "left": "/data/home/luozhiheng/tmp/kwn_pf_cuda_runtime_closure_v1_097a69582ca3_20260901T080254Z/run_r4/restart_qualification/E/continuous_6h.pfzck",
    "right": "/data/home/luozhiheng/tmp/kwn_pf_cuda_runtime_closure_v1_097a69582ca3_20260901T080254Z/run_r4/restart_qualification/E/restart_6h.pfzck",
    "v6_header_and_provenance": {
      "all_fields_exact": true,
      "compared_field_count": 49,
      "mismatched_fields": []
    }
  },
  {
    "auxiliary_populations": {
      "GP": {
        "bin_count": 2,
        "decoded_bins_exact": true,
        "inventory_exact": true,
        "inventory_mol": 1.309926818847619e-21
      },
      "beta_subgrid": {
        "bin_count": 2,
        "decoded_bins_exact": true,
        "inventory_exact": true,
        "inventory_mol": 0.0
      }
    },
    "elastic_warm_state": {
      "byte_count": 10838016,
      "byte_identical": true,
      "present": true
    },
    "fields": {
      "Y": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      },
      "dY_dt_prev": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      },
      "phi": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      },
      "xB": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      }
    },
    "label": "E 24h primary/6h-start checkpoint",
    "left": "/data/home/luozhiheng/tmp/kwn_pf_cuda_runtime_closure_v1_097a69582ca3_20260901T080254Z/run_r4/checkpoints/E/step_87191.pfzck",
    "right": "/data/home/luozhiheng/tmp/kwn_pf_cuda_runtime_closure_v1_097a69582ca3_20260901T080254Z/run_r4/restart_qualification/E/checkpoint_24h_from_6h.pfzck",
    "v6_header_and_provenance": {
      "all_fields_exact": true,
      "compared_field_count": 49,
      "mismatched_fields": []
    }
  },
  {
    "auxiliary_populations": {
      "GP": {
        "bin_count": 2,
        "decoded_bins_exact": true,
        "inventory_exact": true,
        "inventory_mol": 1.309926818847619e-21
      },
      "beta_subgrid": {
        "bin_count": 2,
        "decoded_bins_exact": true,
        "inventory_exact": true,
        "inventory_mol": 0.0
      }
    },
    "elastic_warm_state": {
      "byte_count": 10838016,
      "byte_identical": true,
      "present": true
    },
    "fields": {
      "Y": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      },
      "dY_dt_prev": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      },
      "phi": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      },
      "xB": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      }
    },
    "label": "E 48h primary/6h-continuous",
    "left": "/data/home/luozhiheng/tmp/kwn_pf_cuda_runtime_closure_v1_097a69582ca3_20260901T080254Z/run_r4/checkpoints/E/step_174382.pfzck",
    "right": "/data/home/luozhiheng/tmp/kwn_pf_cuda_runtime_closure_v1_097a69582ca3_20260901T080254Z/run_r4/restart_qualification/E/continuous_6to48h.pfzck",
    "v6_header_and_provenance": {
      "all_fields_exact": true,
      "compared_field_count": 49,
      "mismatched_fields": []
    }
  },
  {
    "auxiliary_populations": {
      "GP": {
        "bin_count": 2,
        "decoded_bins_exact": true,
        "inventory_exact": true,
        "inventory_mol": 1.309926818847619e-21
      },
      "beta_subgrid": {
        "bin_count": 2,
        "decoded_bins_exact": true,
        "inventory_exact": true,
        "inventory_mol": 0.0
      }
    },
    "elastic_warm_state": {
      "byte_count": 10838016,
      "byte_identical": true,
      "present": true
    },
    "fields": {
      "Y": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      },
      "dY_dt_prev": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      },
      "phi": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      },
      "xB": {
        "max_abs_difference": 0.0,
        "within_1e-14": true
      }
    },
    "label": "E 48h 6h-continuous/24h-restart",
    "left": "/data/home/luozhiheng/tmp/kwn_pf_cuda_runtime_closure_v1_097a69582ca3_20260901T080254Z/run_r4/restart_qualification/E/continuous_6to48h.pfzck",
    "right": "/data/home/luozhiheng/tmp/kwn_pf_cuda_runtime_closure_v1_097a69582ca3_20260901T080254Z/run_r4/restart_qualification/E/restart_48h_from_24h.pfzck",
    "v6_header_and_provenance": {
      "all_fields_exact": true,
      "compared_field_count": 49,
      "mismatched_fields": []
    }
  }
]
```

This includes the requested Case A, B and E continuous/restart qualifications and the Case E 6→24→48 h comparison.
