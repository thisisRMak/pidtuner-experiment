"""LQG-flavored counterpart to supervisor_common.py: the priorities
vocabulary and worksheet for the LQR/LQG supervisor.

`FINALIZE_RECOMMENDATION_SCHEMA`/`make_finalize_recommendation_tool` from
supervisor_common.py are domain-agnostic (method_name + rationale, grounded
against whatever names a benchmark tool returned) and are reused as-is by
supervisor_session_lqg.py — no LQG-specific version needed. Only the
priorities side needs its own vocabulary: PID's PRIORITY_CATEGORIES maps to
compare.METRIC_TIERS names (OS%, IAE, Ms/Mt, ...), none of which the LQG
track computes (no MIMO Ms/Mt yet, no overshoot/rise-time outside a
reference-tracking scenario this supervisor doesn't drive — see
docs/lqg_testing.md). This module's categories map onto what
lqg_simulate.compute_regulator_metrics and lqg_checks.py actually return
instead.

Also no `tf_known` field: PID's PrioritiesWorksheet locks tf_known because
it gates which of two competing tools (white-box/black-box) may be called.
The LQG track has one tool, always available (see
docs/lqg_testing.md "Model-following classes" for why there's no black-box
LQG counterpart to gate against) -- so there's nothing to lock.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

# Maps 1:1 onto fields this module's benchmark tool actually returns (see
# supervisor_tools_lqg.py's _serialize_row): the regulator-response metrics
# from lqg_simulate.compute_regulator_metrics, plus pole_margin (a stability-
# margin proxy -- see its docstring -- standing in for the Ms/Mt robustness
# metrics the PID track has and the LQG track doesn't compute yet).
PRIORITY_CATEGORIES_LQG = {
    "speed": ["settling_2pct"],
    "control_effort": ["ISU", "u_peak"],
    "regulation_tightness": ["final_state_norm"],
    "robustness": ["pole_margin"],
}


@dataclass
class LQGPrioritiesWorksheet:
    top_priority: Optional[str] = None
    hard_constraints: list = field(default_factory=list)

    def update(self, top_priority=None, hard_constraints=None):
        if top_priority is not None:
            if top_priority not in PRIORITY_CATEGORIES_LQG:
                return {
                    "ok": False,
                    "error": f"top_priority must be one of {list(PRIORITY_CATEGORIES_LQG)}",
                }
            self.top_priority = top_priority
        if hard_constraints is not None:
            self.hard_constraints = list(hard_constraints)
        return {"ok": True, "worksheet": dataclasses.asdict(self)}


SET_PRIORITIES_LQG_SCHEMA = {
    "type": "function",
    "function": {
        "name": "set_priorities",
        "description": (
            "Record or update the user's priorities worksheet. Call this as "
            "soon as the user states or you infer top_priority or "
            "hard_constraints -- even partially. Fields not provided are left "
            "unchanged."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "top_priority": {
                    "type": "string",
                    "enum": list(PRIORITY_CATEGORIES_LQG.keys()),
                    "description": "The single metric category the user cares about most.",
                },
                "hard_constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Free-text hard constraints, e.g. 'no Kalman filter, full state must be measured'.",
                },
            },
        },
    },
}


def make_set_priorities_lqg_tool(worksheet: LQGPrioritiesWorksheet):
    def _tool(top_priority=None, hard_constraints=None):
        return worksheet.update(top_priority=top_priority, hard_constraints=hard_constraints)

    return _tool
