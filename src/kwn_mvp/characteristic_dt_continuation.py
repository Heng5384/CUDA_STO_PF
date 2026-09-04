"""Local reject-and-halve control for the CR1 characteristic reference.

``CHARACTERISTIC_DT_CONTINUATION_V1`` deliberately does *not* choose a
scalar root when a full characteristic macrostep does not close.  It executes
only direct Picard or an already-qualified exact P2/P4 cycle closure.  A
non-closing candidate is verified to be side-effect free, the complete macro
interval is restored to its immutable accepted start, and that interval is
retried with a uniformly halved local timestep.  The following macro interval
always begins again at the caller's nominal timestep.

The controller is numerical policy, not physical retuning: it does not change
any closure tolerance, Picard cap, growth law, remap, matrix ledger, or
population state outside an accepted substep.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .characteristic_reference import (
    CharacteristicReferenceError,
    CharacteristicReferenceSolver,
    CharacteristicStepDiagnostics,
)
from .ledger import InventorySnapshot
from .solver import RadiusGridOverflowError, SolverConfig


POLICY_NAME = "CHARACTERISTIC_DT_CONTINUATION_V1"
DEFAULT_MAX_REFINEMENT_DEPTH = 6
_ALLOWED_MODES = {"DIRECT", "IDENTITY", "BRACKETED_SCALAR_ROOT"}
_ALLOWED_CYCLE_PERIODS = {2, 4}


class DtContinuationError(RuntimeError):
    """A contract error in local timestep continuation."""


class DtContinuationMinDtError(DtContinuationError):
    """The pre-registered local refinement floor was reached without closure."""


class DtContinuationFatalStepError(DtContinuationError):
    """A non-retryable solver error occurred before a candidate could commit."""


def _hash_arrays(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[key]))
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def accepted_state_hash(solver: CharacteristicReferenceSolver) -> str:
    """Hash all mutable accepted physical state, excluding diagnostic history."""

    return _hash_arrays({key: np.asarray(value) for key, value in solver.state_arrays().items()})


def _history_equal(
    first: Sequence[CharacteristicStepDiagnostics], second: Sequence[CharacteristicStepDiagnostics]
) -> bool:
    return len(first) == len(second) and all(left == right for left, right in zip(first, second))


def _diagnostic_record(diagnostic: CharacteristicStepDiagnostics) -> dict[str, Any]:
    """Keep restart-safe scalar telemetry without serialising solver objects."""

    return {
        "step": int(diagnostic.step),
        "time_s": float(diagnostic.time_s),
        "dt_s": float(diagnostic.dt_s),
        "matrix_xb": float(diagnostic.matrix_xb),
        "fixed_point_iterations": int(diagnostic.fixed_point_iterations),
        "fixed_point_picard_iterations": int(diagnostic.fixed_point_picard_iterations),
        "fixed_point_xb_residual": float(diagnostic.fixed_point_xb_residual),
        "fixed_point_xb_tolerance": float(diagnostic.fixed_point_xb_tolerance),
        "fixed_point_population_residual": float(diagnostic.fixed_point_population_residual),
        "fixed_point_cell_measure_residual": float(diagnostic.fixed_point_cell_measure_residual),
        "fixed_point_convergence_mode": str(diagnostic.fixed_point_convergence_mode),
        "fixed_point_periodic_cycle_period": int(diagnostic.fixed_point_periodic_cycle_period),
        "fixed_point_bracketed_root_iterations": int(diagnostic.fixed_point_bracketed_root_iterations),
        "fixed_point_root_verification_kind": str(diagnostic.fixed_point_root_verification_kind),
        "inventory_relative_residual": float(diagnostic.inventory.relative_residual),
        "Q_total_mol_m3": float(diagnostic.inventory.total_mol_m3),
        "Q_beta_mol_m3": float(diagnostic.inventory.beta_resolved_mol_m3),
        "Q_matrix_mol_m3": float(diagnostic.inventory.matrix_mol_m3),
    }


def _diagnostic_payload(diagnostic: CharacteristicStepDiagnostics) -> dict[str, Any]:
    """Serialise an accepted diagnostic exactly enough to restore history.

    The base CR1 checkpoint intentionally stores physical state rather than
    its potentially long telemetry history.  An in-macro continuation restart
    must also preserve that history, because its closure modes and accepted
    residuals are part of the policy audit.  The sidecar is therefore the
    narrow, policy-owned place where it is retained.
    """

    return dict(asdict(diagnostic))


def _diagnostic_from_payload(payload: Mapping[str, Any]) -> CharacteristicStepDiagnostics:
    """Validate and reconstruct one sidecar diagnostic without approximation."""

    expected = {item.name for item in fields(CharacteristicStepDiagnostics)}
    values = dict(payload)
    if set(values) != expected:
        raise DtContinuationError("continuation restart diagnostic schema differs")
    inventory = values.get("inventory")
    if not isinstance(inventory, Mapping):
        raise DtContinuationError("continuation restart diagnostic inventory differs")
    inventory_fields = {item.name for item in fields(InventorySnapshot)}
    if set(inventory) != inventory_fields:
        raise DtContinuationError("continuation restart inventory schema differs")
    values["inventory"] = InventorySnapshot(**dict(inventory))
    try:
        return CharacteristicStepDiagnostics(**values)
    except (TypeError, ValueError) as error:
        raise DtContinuationError("continuation restart diagnostic is invalid") from error


@dataclass(frozen=True)
class SolverStateSnapshot:
    """An in-memory accepted state used only to roll back an unclosed macrostep."""

    arrays: Mapping[str, np.ndarray]
    history: tuple[CharacteristicStepDiagnostics, ...]
    state_hash: str

    @classmethod
    def capture(cls, solver: CharacteristicReferenceSolver) -> "SolverStateSnapshot":
        arrays = {key: np.asarray(value).copy() for key, value in solver.state_arrays().items()}
        return cls(arrays=arrays, history=tuple(solver.history), state_hash=_hash_arrays(arrays))

    def matches(self, solver: CharacteristicReferenceSolver) -> bool:
        current = solver.state_arrays()
        arrays_equal = set(current) == set(self.arrays) and all(
            np.array_equal(np.asarray(current[key]), np.asarray(self.arrays[key])) for key in self.arrays
        )
        return bool(
            arrays_equal
            and accepted_state_hash(solver) == self.state_hash
            and _history_equal(tuple(solver.history), self.history)
        )

    def restore(self, solver: CharacteristicReferenceSolver) -> None:
        required = {
            "g_number_density_per_m4",
            "beta_number_density_per_m4",
            "matrix_xb",
            "time_s",
            "step",
            "cumulative_number_dissolution_m3",
            "cumulative_beta_volume_dissolution",
            "cumulative_mol_b_returned_mol_m3",
        }
        if set(self.arrays) != required:
            raise DtContinuationError("continuation snapshot has an unexpected CR1 state schema")
        solver.population("g").number_density_per_m4[:] = np.asarray(
            self.arrays["g_number_density_per_m4"], dtype=np.float64
        )
        solver.population("beta").number_density_per_m4[:] = np.asarray(
            self.arrays["beta_number_density_per_m4"], dtype=np.float64
        )
        solver.matrix_xb = float(np.asarray(self.arrays["matrix_xb"])[0])
        solver.time_s = float(np.asarray(self.arrays["time_s"])[0])
        solver.step = int(np.asarray(self.arrays["step"])[0])
        solver.cumulative_number_dissolution_m3 = float(
            np.asarray(self.arrays["cumulative_number_dissolution_m3"])[0]
        )
        solver.cumulative_beta_volume_dissolution = float(
            np.asarray(self.arrays["cumulative_beta_volume_dissolution"])[0]
        )
        solver.cumulative_mol_b_returned_mol_m3 = float(
            np.asarray(self.arrays["cumulative_mol_b_returned_mol_m3"])[0]
        )
        solver.history = list(self.history)
        if not self.matches(solver):
            raise DtContinuationError("continuation rollback did not restore the accepted state exactly")


@dataclass(frozen=True)
class ContinuationMacroResult:
    """The finally accepted subdivision of one nominal physical macro interval."""

    macro_index: int
    start_time_s: float
    end_time_s: float
    nominal_dt_s: float
    refinement_depth: int
    accepted_diagnostics: tuple[CharacteristicStepDiagnostics, ...]
    rejection_count: int
    state_hash: str


@dataclass
class _ActiveMacro:
    macro_index: int
    start_time_s: float
    end_time_s: float
    nominal_dt_s: float
    refinement_depth: int
    pending: list[tuple[float, int]]
    start_snapshot: SolverStateSnapshot | None
    start_checkpoint: Path | None
    solver_class: type[CharacteristicReferenceSolver]
    start_history: tuple[CharacteristicStepDiagnostics, ...]
    accepted_diagnostics: list[CharacteristicStepDiagnostics] = field(default_factory=list)
    rejection_count: int = 0


def _is_retryable_closure_error(error: CharacteristicReferenceError) -> bool:
    """Accept only an explicit ordinary-closure nonconvergence for halving.

    Inventory corruption, invalid traces, non-finite fields, and other solver
    contract failures must remain fatal rather than being concealed as local
    stiffness.  The frozen CR1 nonconvergence message is deliberately precise.
    """

    message = str(error)
    return message.startswith("characteristic matrix/population fixed point did not converge after") or (
        message.startswith("exact fixed-point period-")
        and ("did not close" in message or "did not provide" in message)
    )


class CharacteristicDtContinuation:
    """Run local, reject-and-halve CR1 macro intervals with no scalar fallback."""

    def __init__(
        self,
        solver: CharacteristicReferenceSolver,
        *,
        max_refinement_depth: int = DEFAULT_MAX_REFINEMENT_DEPTH,
    ) -> None:
        if (
            isinstance(max_refinement_depth, bool)
            or not isinstance(max_refinement_depth, int)
            or max_refinement_depth < 0
        ):
            raise ValueError("max_refinement_depth must be a non-negative integer")
        self.solver = solver
        self.max_refinement_depth = int(max_refinement_depth)
        self.events: list[dict[str, Any]] = []
        self._next_macro_index = 1
        self._active: _ActiveMacro | None = None

    @property
    def active_macro(self) -> _ActiveMacro | None:
        """Expose read-only active-macro presence for checkpoint orchestration."""

        return self._active

    def _validate_accepted_diagnostic(self, diagnostic: CharacteristicStepDiagnostics) -> None:
        mode = str(diagnostic.fixed_point_convergence_mode)
        period = int(diagnostic.fixed_point_periodic_cycle_period)
        if mode not in _ALLOWED_MODES:
            raise DtContinuationError(
                f"{POLICY_NAME} forbids closure mode {mode}; no scalar-root fallback is legal"
            )
        if mode == "BRACKETED_SCALAR_ROOT" and period not in _ALLOWED_CYCLE_PERIODS:
            raise DtContinuationError("only exact P2/P4 cycle closures are qualified in continuation")
        if mode != "BRACKETED_SCALAR_ROOT" and period != 0:
            raise DtContinuationError("direct continuation closure reported an unexpected cycle period")
        if int(diagnostic.fixed_point_bracketed_root_iterations) > 0 and mode != "BRACKETED_SCALAR_ROOT":
            raise DtContinuationError("continuation direct step unexpectedly attempted a scalar root")
        if mode == "BRACKETED_SCALAR_ROOT" and int(diagnostic.fixed_point_bracketed_root_iterations) <= 0:
            raise DtContinuationError("qualified P2/P4 closure omitted its exact-cycle root telemetry")
        scalar_residual = float(diagnostic.fixed_point_xb_residual)
        scalar_tolerance = float(diagnostic.fixed_point_xb_tolerance)
        population_residual = float(diagnostic.fixed_point_population_residual)
        cell_measure_residual = float(diagnostic.fixed_point_cell_measure_residual)
        if (
            not math.isfinite(scalar_residual)
            or not math.isfinite(scalar_tolerance)
            or scalar_residual > scalar_tolerance
            or not math.isfinite(population_residual)
            or population_residual > self.solver._population_convergence_rtol
            or not math.isfinite(cell_measure_residual)
        ):
            raise DtContinuationError("accepted continuation step failed the frozen fixed-point acceptance")
        inventory_residual = float(diagnostic.inventory.relative_residual)
        if not math.isfinite(inventory_residual) or inventory_residual > float(
            self.solver.config.inventory_tolerance_relative
        ):
            raise DtContinuationError("accepted continuation step failed the frozen inventory tolerance")
        cells = np.asarray(self.solver._beta_cell_numbers(), dtype=np.float64)
        if not np.all(np.isfinite(cells)) or np.any(cells < 0.0):
            raise DtContinuationError("accepted continuation step has invalid beta cell numbers")

    def begin_macro(
        self,
        nominal_dt_s: float,
        *,
        macro_start_checkpoint: str | Path | None = None,
    ) -> None:
        """Start one macro interval; callers may pause after any accepted leaf."""

        if self._active is not None:
            raise DtContinuationError("a continuation macro interval is already active")
        dt = float(nominal_dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("nominal_dt_s must be finite and positive")
        start_snapshot = SolverStateSnapshot.capture(self.solver)
        checkpoint = None if macro_start_checkpoint is None else Path(macro_start_checkpoint)
        if checkpoint is not None:
            if checkpoint.exists():
                raise DtContinuationError("refusing to overwrite a macro-start checkpoint")
            self.solver.save_checkpoint(checkpoint)
        self._active = _ActiveMacro(
            macro_index=self._next_macro_index,
            start_time_s=float(self.solver.time_s),
            end_time_s=float(self.solver.time_s + dt),
            nominal_dt_s=dt,
            refinement_depth=0,
            pending=[(dt, 0)],
            start_snapshot=start_snapshot,
            start_checkpoint=checkpoint,
            solver_class=type(self.solver),
            start_history=tuple(self.solver.history),
        )
        self._next_macro_index += 1

    def _restore_macro_start(self, active: _ActiveMacro) -> None:
        if active.start_snapshot is not None:
            active.start_snapshot.restore(self.solver)
            return
        if active.start_checkpoint is None:
            raise DtContinuationError("active resumed macro has no immutable rollback state")
        restored = active.solver_class.load_checkpoint(
            config=self.solver.config, path=active.start_checkpoint
        )
        restored.history = list(active.start_history)
        self.solver = restored
        if self.solver.time_s != active.start_time_s:
            raise DtContinuationError("macro-start checkpoint time differs from its continuation sidecar")

    def _restart_at_next_depth(self, active: _ActiveMacro) -> None:
        if active.refinement_depth >= self.max_refinement_depth:
            self._restore_macro_start(active)
            self._active = None
            raise DtContinuationMinDtError(
                f"{POLICY_NAME} reached dt/{2 ** active.refinement_depth} without an admissible closure"
            )
        previous_depth = active.refinement_depth
        for event in self.events:
            if (
                event.get("event") == "ACCEPTED_SUBSTEP"
                and int(event.get("macro_index", -1)) == active.macro_index
                and not bool(event.get("rolled_back", False))
            ):
                event["rolled_back"] = True
        self._restore_macro_start(active)
        active.refinement_depth += 1
        pieces = 2 ** active.refinement_depth
        leaf_dt = active.nominal_dt_s / pieces
        active.pending = [(leaf_dt, active.refinement_depth) for _ in range(pieces)]
        active.accepted_diagnostics.clear()
        self.events.append(
            {
                "event": "MACRO_RESTART_HALVED",
                "macro_index": active.macro_index,
                "macro_start_time_s": active.start_time_s,
                "nominal_dt_s": active.nominal_dt_s,
                "previous_refinement_depth": previous_depth,
                "refinement_depth": active.refinement_depth,
                "leaf_dt_s": leaf_dt,
                "accepted_state_hash_after_rollback": accepted_state_hash(self.solver),
            }
        )

    def advance_pending(self) -> ContinuationMacroResult | None:
        """Execute one scheduled leaf; return a result only when the macro closes."""

        active = self._active
        if active is None:
            raise DtContinuationError("no continuation macro interval is active")
        dt_s, depth = active.pending[0]
        trial_snapshot = SolverStateSnapshot.capture(self.solver)
        before_hash = trial_snapshot.state_hash
        try:
            diagnostic = self.solver.advance_one_ordinary_or_qualified_cycle(maximum_dt_s=dt_s)
        except RadiusGridOverflowError as error:
            unchanged = trial_snapshot.matches(self.solver)
            self._restore_macro_start(active)
            self._active = None
            if not unchanged:
                raise DtContinuationError(
                    "non-retryable Rmax overflow mutated accepted CR1 state before rollback"
                ) from error
            raise DtContinuationFatalStepError(
                f"non-retryable Rmax overflow at continuation depth {depth}; state_unchanged=true: {error}"
            ) from error
        except CharacteristicReferenceError as error:
            after_hash = accepted_state_hash(self.solver)
            history_unchanged = trial_snapshot.matches(self.solver)
            if not history_unchanged:
                self._restore_macro_start(active)
                self._active = None
                raise DtContinuationError("a rejected ordinary closure mutated accepted CR1 state") from error
            if not _is_retryable_closure_error(error):
                self._restore_macro_start(active)
                self._active = None
                raise DtContinuationFatalStepError(
                    f"non-retryable characteristic failure at continuation depth {depth}: {error}"
                ) from error
            active.rejection_count += 1
            self.events.append(
                {
                    "event": "REJECTED_NONCLOSING_SUBSTEP",
                    "macro_index": active.macro_index,
                    "macro_start_time_s": active.start_time_s,
                    "nominal_dt_s": active.nominal_dt_s,
                    "candidate_dt_s": dt_s,
                    "candidate_depth": depth,
                    "accepted_state_hash_before_trial": before_hash,
                    "accepted_state_hash_after_rejection": after_hash,
                    "history_unchanged": history_unchanged,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            self._restart_at_next_depth(active)
            return None
        if float(diagnostic.dt_s) != dt_s:
            self._restore_macro_start(active)
            self._active = None
            raise DtContinuationError(
                "continuation solver accepted a dt different from the scheduled local substep"
            )
        try:
            self._validate_accepted_diagnostic(diagnostic)
        except DtContinuationError:
            self._restore_macro_start(active)
            self._active = None
            raise
        active.pending.pop(0)
        active.accepted_diagnostics.append(diagnostic)
        self.events.append(
            {
                "event": "ACCEPTED_SUBSTEP",
                "macro_index": active.macro_index,
                "macro_start_time_s": active.start_time_s,
                "nominal_dt_s": active.nominal_dt_s,
                "effective_dt_s": dt_s,
                "refinement_depth": depth,
                "rolled_back": False,
                "state_hash_after_accept": accepted_state_hash(self.solver),
                **_diagnostic_record(diagnostic),
            }
        )
        if active.pending:
            return None
        if self.solver.time_s != active.end_time_s:
            self._restore_macro_start(active)
            self._active = None
            raise DtContinuationError("accepted local subdivision did not land on its macro endpoint exactly")
        result = ContinuationMacroResult(
            macro_index=active.macro_index,
            start_time_s=active.start_time_s,
            end_time_s=active.end_time_s,
            nominal_dt_s=active.nominal_dt_s,
            refinement_depth=active.refinement_depth,
            accepted_diagnostics=tuple(active.accepted_diagnostics),
            rejection_count=active.rejection_count,
            state_hash=accepted_state_hash(self.solver),
        )
        self._active = None
        return result

    def run_active_macro(self) -> ContinuationMacroResult:
        """Finish an already prepared macro interval without changing its nominal dt."""

        while True:
            result = self.advance_pending()
            if result is not None:
                return result

    def advance_macro(
        self,
        nominal_dt_s: float,
        *,
        macro_start_checkpoint: str | Path | None = None,
    ) -> ContinuationMacroResult:
        """Run one macro interval, locally refining only this interval if necessary."""

        self.begin_macro(nominal_dt_s, macro_start_checkpoint=macro_start_checkpoint)
        return self.run_active_macro()

    def save_active_restart_bundle(
        self,
        *,
        checkpoint_path: str | Path,
        sidecar_path: str | Path,
    ) -> dict[str, Any]:
        """Persist an accepted substep boundary inside a locally refined macro.

        A CR1 solver checkpoint alone cannot encode the pending half-step queue.
        This sidecar binds that queue, the current checkpoint, and the
        immutable macro-start checkpoint needed if a later leaf must force a
        deeper rollback.
        """

        active = self._active
        if active is None or active.start_checkpoint is None:
            raise DtContinuationError("an active macro with a saved start checkpoint is required")
        checkpoint = Path(checkpoint_path)
        sidecar = Path(sidecar_path)
        if checkpoint.exists() or sidecar.exists():
            raise DtContinuationError("refusing to overwrite a continuation restart artifact")
        if not active.accepted_diagnostics:
            raise DtContinuationError("restart inside a macro requires at least one accepted substep")
        self.solver.save_checkpoint(checkpoint)
        payload = {
            "policy": POLICY_NAME,
            "max_refinement_depth": self.max_refinement_depth,
            "current_checkpoint": str(checkpoint),
            "current_checkpoint_sha256": _sha256_file(checkpoint),
            "current_state_hash": accepted_state_hash(self.solver),
            "source_config_hash": self.solver.config.source_config_hash,
            "validation_contract_hash": self.solver.contract_hash,
            "macro_start_checkpoint": str(active.start_checkpoint),
            "macro_start_checkpoint_sha256": _sha256_file(active.start_checkpoint),
            "macro_index": active.macro_index,
            "macro_start_time_s": active.start_time_s,
            "macro_end_time_s": active.end_time_s,
            "nominal_dt_s": active.nominal_dt_s,
            "refinement_depth": active.refinement_depth,
            "pending": [{"dt_s": dt_s, "depth": depth} for dt_s, depth in active.pending],
            "rejection_count": active.rejection_count,
            "events": self.events,
            "macro_start_history": [_diagnostic_payload(item) for item in active.start_history],
            "accepted_diagnostics": [
                _diagnostic_payload(item) for item in active.accepted_diagnostics
            ],
        }
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"checkpoint": str(checkpoint), "sidecar": str(sidecar), **payload}

    @classmethod
    def load_active_restart_bundle(
        cls,
        *,
        config: SolverConfig,
        sidecar_path: str | Path,
        solver_class: type[CharacteristicReferenceSolver] = CharacteristicReferenceSolver,
    ) -> "CharacteristicDtContinuation":
        """Restore an accepted in-macro boundary after validating its sidecar."""

        sidecar = Path(sidecar_path)
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise DtContinuationError("continuation restart sidecar is invalid") from error
        if payload.get("policy") != POLICY_NAME:
            raise DtContinuationError("continuation restart policy differs")
        if payload.get("source_config_hash") != config.source_config_hash:
            raise DtContinuationError("continuation restart config differs")
        checkpoint = Path(str(payload.get("current_checkpoint", "")))
        start_checkpoint = Path(str(payload.get("macro_start_checkpoint", "")))
        if not checkpoint.is_file() or not start_checkpoint.is_file():
            raise DtContinuationError("continuation restart checkpoint is unavailable")
        if _sha256_file(checkpoint) != payload.get("current_checkpoint_sha256"):
            raise DtContinuationError("continuation current checkpoint hash differs")
        if _sha256_file(start_checkpoint) != payload.get("macro_start_checkpoint_sha256"):
            raise DtContinuationError("continuation macro-start checkpoint hash differs")
        solver = solver_class.load_checkpoint(config=config, path=checkpoint)
        if solver.contract_hash != payload.get("validation_contract_hash"):
            raise DtContinuationError("continuation restart contract hash differs")
        if accepted_state_hash(solver) != payload.get("current_state_hash"):
            raise DtContinuationError("continuation restart accepted-state hash differs")
        controller = cls(solver, max_refinement_depth=int(payload["max_refinement_depth"]))
        pending = [(float(item["dt_s"]), int(item["depth"])) for item in payload.get("pending", [])]
        if not pending:
            raise DtContinuationError("continuation restart has no pending substep")
        depth = int(payload["refinement_depth"])
        if depth < 0 or depth > controller.max_refinement_depth or any(item_depth != depth for _, item_depth in pending):
            raise DtContinuationError("continuation restart pending-depth contract differs")
        try:
            start_history = tuple(
                _diagnostic_from_payload(item) for item in payload.get("macro_start_history", [])
            )
            accepted_diagnostics = [
                _diagnostic_from_payload(item) for item in payload.get("accepted_diagnostics", [])
            ]
        except TypeError as error:
            raise DtContinuationError("continuation restart history is invalid") from error
        if len(accepted_diagnostics) >= 1:
            if accepted_diagnostics[-1].step != solver.step or accepted_diagnostics[-1].time_s != solver.time_s:
                raise DtContinuationError("continuation restart accepted history differs from checkpoint")
        elif solver.step != len(start_history):
            raise DtContinuationError("continuation restart lacks the required accepted in-macro history")
        solver.history = list(start_history) + accepted_diagnostics
        controller.events = [dict(item) for item in payload.get("events", [])]
        controller._next_macro_index = int(payload["macro_index"]) + 1
        controller._active = _ActiveMacro(
            macro_index=int(payload["macro_index"]),
            start_time_s=float(payload["macro_start_time_s"]),
            end_time_s=float(payload["macro_end_time_s"]),
            nominal_dt_s=float(payload["nominal_dt_s"]),
            refinement_depth=depth,
            pending=pending,
            start_snapshot=None,
            start_checkpoint=start_checkpoint,
            solver_class=solver_class,
            start_history=start_history,
            accepted_diagnostics=accepted_diagnostics,
            rejection_count=int(payload["rejection_count"]),
        )
        return controller


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
