# Provenance

This stage is read-only and does not start PF, modify a checkpoint, or run a PSD scan.

- branch: `codex/pf-dynamic-microstructure-audit-v1`
- commit: `1c08f9ee011b31e0cd4d82749e58a8f69ebd2204`
- committed source-tree digest: `a27ad86e72d92beb9adc7fbcace59789d853c823f66767ac94b1d19c8e2b082b`
- production CUDA binary SHA-256: `516489b3e4dbafd6ba5876beb2858df8309fbfcbd1065d455b73f1782f5fe8f5`

## Inputs

- analysis_script: `e35789ab77f8ca4e090e3e3673658408c85d87414a050a6423a388b0a4bf3442` (`/Users/heng/Documents/GitHub/CUDA_STO_PF/scripts/reproduce_sheskin_pf_density_baseline_v1.py`)
- digitization_1: `c4c395983bd999a237d4195dafb3d110b975fd2600651011b67169a70accfe51` (`/Users/heng/Documents/GitHub/CUDA_STO_PF/tmp/sheskin_stage1_run1/sheskin_experimental_kappa.csv`)
- digitization_2: `bf2d963200559e8b30e5785365ef285ba880943f879fd67622079d45ddbde93f` (`/Users/heng/Documents/GitHub/CUDA_STO_PF/tmp/sheskin_stage1_run1/sheskin_figure5c_calibration.csv`)
- digitization_3: `04f681e8b272fcd71305f1164837e8c0e0c20379e00fe2cc83edb5ffa0c62934` (`/Users/heng/Documents/GitHub/CUDA_STO_PF/tmp/sheskin_stage1_run1/sheskin_figure5c_digitized_pixels.csv`)
- sheskin_main_pdf: `fa159ef4b24568bc168bfe6859f496122a3d599de8c822664f635fa4cd62d0c4` (`/Users/heng/Library/CloudStorage/OneDrive-GuangdongTechnion-IsraelInstituteofTechnology/Project/Diffusion/Paper/tailoring-thermoelectric-transport-properties-of-ag-alloyed-pbte-effects-of-microstructure-evolution.pdf`)
- transport_parameter_contract: `d16e948dd34403129b2deecb90efbea937251e23ae9c128c353b6f66bbee1ae0` (`/Users/heng/Documents/GitHub/CUDA_STO_PF/data/qualification/pf_full_psd_no_dislocation_transport_v1/transport_parameter_contract.json`)
- transport_script: `a42f933c43d0adbd4fc9a4d86a7b74bc08d4efe64517cc3fbf1ba177b62dcf66` (`/Users/heng/Documents/GitHub/CUDA_STO_PF/scripts/pf_full_psd_no_dislocation_transport_v1.py`)
- yu_main_pdf: `da838898f561a26b00b37124b18f18df627adf95bf9fe4622f3089f3aaa19eba` (`/Users/heng/Library/CloudStorage/OneDrive-GuangdongTechnion-IsraelInstituteofTechnology/Project/Diffusion/Paper/Advanced Energy Materials - 2024 - Yu - Ostwald Ripening of Ag2Te Precipitates in Thermoelectric PbTe  Effects of.pdf`)
- yu_parameter_contract: `163562fa19f0cbcf3731f181ecae6c1d34f4b5952cdab9a81a5aa3eabd9f1c8f` (`/Users/heng/Documents/GitHub/CUDA_STO_PF/data/qualification/yu2024_transport_v1/yu_48h_parameters.json`)
- yu_supporting_information_pdf: `f60bef889c0445b790f9a93828aa40b7ca56bf40850a461b0aff019a05d37ff0` (`/Users/heng/Library/CloudStorage/OneDrive-GuangdongTechnion-IsraelInstituteofTechnology/Project/Diffusion/Paper/AEM Yu 2024 SM.pdf`)

## A/B/C authorities

- A: audit `dfd0ad7824fe66acd81ba0f788399a801f3055402f57d0d5f9a8ad89e9688ef4`, fixture `f6cce1bd1abaf0e8767f52fc70009a1cce9c928a442bc3ebd38e7c8e7c26d7b2`, merge-aware audit `9f487839ec75c626577e72b180ed9435bb75080022ea8cc7c5e928fa29e41d98`, checkpoint-chain `5daefa3b483583288bc79207c37e9b3144b5f1d27ae77d0af5fe18f6a0c377c3` (44 entries)
- B: audit `d0fd7e947f47935dac4a0823c05b6069e74cd0f998d1b63b3c76df47c8aeac8d`, fixture `8106091d92b2924f65a23ab5adf8e98d958dd6bac63f3fb099c4f85567c40389`, merge-aware audit `10185625c90f18d2336515b90b3ac8e21ea1efbc3b464b5a9cacb4342e83c95a`, checkpoint-chain `b794ad85fc891ea0a2b6a47dc3be86f810c421de7f27f97cf3c5292e8d0cb967` (44 entries)
- C: audit `5b22ef813ada4131b4a5760909f528c3981c65b8aaf9268b523f9e1045d2046b`, fixture `2ad665a3b58d26c29fcfdbfc2c949be979e602147283fce25bc00d992acaac64`, merge-aware audit `7e22455465f3d8c7661673286ef9cdb47d5e1cb46ee319d3513e13d66cb5f3d6`, checkpoint-chain `26d022eaae8e8ffe953c1877d77826a5c3988d6114ebc9b13d515fc41f6f26f8` (44 entries)

## Accepted-field replay and final analysis chain

The production binary above remains the immutable PF authority binary. The separate workstation replay binary only adds a diagnostic stop/output path; it does not write an authority checkpoint or advance PF time.

- replay CUDA source: `d4809ca43e04034eeaa3df92df3dc93a0a196530ac34014aad850bedf3d20453`
- replay workstation binary: `68f94f3128c73c5f9316543b4234b35ab00b13638983bdf7223b976b33f3be36`
- frozen production-physics parameter file: `ecbdd0ac070bdf5e5d214322b5248a08f5ca5dd4e0670427513f5ef977ea977a`
- replay qualification analysis: `1837c2982043285c80209967057466cafc07d7ed631f6406c2736549f05edb23`
- historical replay assembler: `3f611711cf8377318aa135f2ae5a2da854d1f07b230c5143a1d433232bb53c13`
- periodic interface/strain descriptor analysis: `725a4ced52a1852f74578c21a40662cd00db386b8b39e68577a1c4898b0614c5`
- leverage analysis: `a201d27682917aa85e2b21c660320296834bde25199415b654c179edc58140fb`
- AQ/6 h freeze and 48 h blind evaluator: `518215777408dcc9c02d18beb05a6131eb003856bbd8039e5a55a0ebe292d13b`
- final fail-closed adjudicator: `6c43050fcf89c820084720c5978c440c1650bcecea83f233df6cf6406310529a`
- frozen-before-48 h manifest: `8e7dfa0259830ab73c81ef10410c3505b85a0cf81d7d3e27ab42c94562716234`
- completion audit: `05cf158dfb72775b09da3810282018576983cfd3dfaa7bd507179fe4904d8421`

Full accepted-field hashes, selected checkpoint hashes, solver residuals, descriptor-source hashes, frozen model inputs, output ledgers, and the preserved pre-adjudication status-label evidence are stored inside this report directory. Two independent descriptor runs and two independent calibration/blind runs were byte-identical. No commit or push was performed.
