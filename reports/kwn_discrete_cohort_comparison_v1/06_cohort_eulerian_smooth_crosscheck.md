# Cohort–Eulerian smooth-PSD crosscheck

Status: `FAIL_COHORT_EULERIAN_PHYSICS_MISMATCH`.  The terminal 3200-node deterministic quadrature is the cohort comparison authority. 1600→3200 quadrature refinement: `True`; global direction agreement: `True`; 2% continuous-metric gate (including cumulative dissolution inventory): `False`.  Requalified-authority binding: `True`.

The crosscheck fails on: `M0_m3` at 48 h (2.43208%), `N_m0_m3` at 48 h (2.43208%), `Rmean3_m3` at 48 h (2.39603%), `cumulative_dissolution_inventory_mol_m3` at 0.1 h (21.2187%).

All deterministic cohort representations use the same frozen beta growth law, curvature equilibrium, D(T), molar volumes, total inventory and matrix inverse as Eulerian KWN.  Differences are therefore reported as representation differences, not physical retuning.
