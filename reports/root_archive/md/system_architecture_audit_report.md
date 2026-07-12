# System Architecture Audit Report

## Executive Verdict

The current Ag2Te-PbTe GP -> beta nucleation system is a **hybrid, partly non-variational multi-layer model**.

It contains a substantial phase-field backbone and several physically motivated CNT/event operators, but the four requested layers are not coupled through one fully consistent thermodynamic functional. GP-to-beta nucleation is implemented through a mixture of PF evolution, stochastic CNT-like operators, scheduled insertion, and direct event injection.

Final classification:

```text
D. Inconsistent multi-layer system
```

Operationally, it behaves like a **kinetic event-driven hybrid PF + CNT system**, but the duplicated GP and nucleation routes make it too internally mixed to classify as a clean physically consistent hybrid model.

## Architecture Diagram

```text
Thermodynamic utilities / CALPHAD-like chemical potentials
  |
  v
Composition transport backbone
  - xB / Y / xBtot storage
  - phase-weighted mobility
  - flux-divergence update
  |
  +------------------------------+
  |                              |
  v                              v
Beta PF field phi            GP PF field eta
  - double well W              - GP double well gp_W_eta
  - gradient kappa_phi         - gradient gp_kappa_eta
  - chemical/elastic drive     - chemical/surrogate/elastic drive
  |                              |
  +---------------+--------------+
                  |
                  v
        Alpha / GP / beta phase fractions
        h_alpha, h_GP, h_beta
                  |
                  v
        Storage, mobility, composition feedback

Parallel nucleation/event layer
  |
  +--> scheduled beta insertion
  +--> GP-assisted beta reservoir events
  +--> GP-to-beta stochastic CNT-like conversion
  +--> GP nucleation event seeding
  +--> selector/template-driven insertion path
                  |
                  v
        beta seed inserted into phi/xB/eta fields
                  |
                  v
        PF growth and transport after insertion
```

## Layer Verification

| Layer | Status | Reason |
|---|---:|---|
| 1. PF microstructure evolution backbone | partial | Composition transport, beta PF, GP PF, mobility, and chemical potentials exist. However composition-gradient energy is not clearly implemented as a first-class `kappa_x |grad x|^2` term, and nucleation can bypass PF variational dynamics. |
| 2. GP zone evolution + morphology | partial | Dynamic eta GP morphology exists, but GP thermodynamics are surrogate/parameterized and duplicated by a separate reservoir/event GP module. |
| 3. Beta phase nucleation + growth | partial | Beta growth is continuous PF; beta nucleation is mainly event inserted or CNT-like stochastic. |
| 4. GP-modulated heterogeneous nucleation | partial | GP affects beta events through eta detection, hazard/barrier modulation, and direct reservoir injection. These routes are not unified. |

## Missing Physics Components

- A single free-energy functional that consistently includes alpha, GP, beta, composition, elastic, and interface terms.
- A clearly documented composition-gradient energy term for `x_B`.
- One canonical GP-to-beta heterogeneous nucleation operator.
- One canonical `S(x)` or barrier-modulation field used by all nucleation paths.
- Unified treatment of continuous GP eta and GP reservoir/site events.
- Quantitative validation that CNT barrier, selector output, discrete template insertion, and observed CUDA event statistics agree.

## Duplicated / Inconsistent Mechanisms

| Mechanism | Duplicated Paths | Consequence |
|---|---|---|
| GP representation | Continuous eta field and GP site/reservoir module | GP can affect beta through unrelated physics routes. |
| Beta nucleation | Scheduled insertion, GP-assisted reservoir event, GP-to-beta conversion, stochastic hazard | Event source can determine outcome as much as physics. |
| Barrier logic | CNT-like formula, hazard barrier, selector/catalog barrier, scheduled event timing | No single barrier controls all insertion decisions. |
| Coupling route | Variational-like PF, stochastic CNT, direct event injection | Coupling is mixed, not fully energy-consistent. |

## GP -> Beta Coupling Type

```text
mixed: variational-like + stochastic + direct event injection
```

The coupling is not purely variational. It is also not purely stochastic, because eta and phi fields have real PF dynamics. The correct description is a mixed hybrid architecture with inconsistent parallel routes.

## Publication Readiness

| Paper Framing | Readiness | Rationale |
|---|---:|---|
| Full PF paper | no | Event insertion and GP reservoir paths break a fully variational interpretation. |
| CNT paper | no | CNT-like operators are present but not uniquely controlling all nucleation events. |
| Hybrid nucleation paper | conditional/no | The system could support a hybrid PF+CNT paper after disabling bypass paths or explicitly validating them. Current architecture still has duplicated mechanisms and incomplete physics closure. |

## Final Classification

```text
system_classification = D
```

The current system is best described as an **inconsistent multi-layer system** with a strong PF backbone and useful hybrid CNT/event capabilities. It is not yet a physics-consistent closed-loop GP -> beta nucleation engine.
