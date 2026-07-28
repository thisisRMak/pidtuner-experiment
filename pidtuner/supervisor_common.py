"""Supervisor building blocks shared by both the white-box and black-box tool
families: the priorities worksheet, the priority-category vocabulary, and the
two tools (`set_priorities`, `finalize_recommendation`) that never need to
touch a plant or a signal.

No imports from plant.py, signal_format.py, or blackbox.py — this module is
safe for both tool families to depend on.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

# The vocabulary the LLM and user negotiate over maps 1:1 onto the real
# metric names in compare.METRIC_TIERS, grouped the way a non-control-
# engineer would talk about them. No taxonomy is invented that doesn't
# correspond to a metric a benchmark tool can actually return.
PRIORITY_CATEGORIES = {
    "speed": ["Rise", "ts"],
    "overshoot": ["OS%"],
    "tracking_accuracy": ["IAE"],
    "disturbance_rejection": ["IAE_load"],
    "robustness": ["Ms", "Mt", "GM_dB", "PM_deg"],
    "control_effort": ["ISU", "u_tv", "u_peak"],
}


@dataclass
class PrioritiesWorksheet:
    """The minimal "worksheet" the supervisor elicits from the user.

    tf_known is locked once set (see `update`) so a session can never
    silently flip between the white-box and black-box benchmark tool
    mid-conversation, which would blur the provenance of already-cited
    numbers.
    """

    tf_known: Optional[bool] = None
    top_priority: Optional[str] = None
    hard_constraints: list = field(default_factory=list)

    def update(self, tf_known=None, top_priority=None, hard_constraints=None):
        if tf_known is not None:
            if self.tf_known is not None and self.tf_known != tf_known:
                return {
                    "ok": False,
                    "error": (
                        f"tf_known is already set to {self.tf_known} for this "
                        "session and cannot be changed. Start a new session "
                        "(/reset) if the mode genuinely needs to change."
                    ),
                }
            self.tf_known = tf_known
        if top_priority is not None:
            if top_priority not in PRIORITY_CATEGORIES:
                return {
                    "ok": False,
                    "error": f"top_priority must be one of {list(PRIORITY_CATEGORIES)}",
                }
            self.top_priority = top_priority
        if hard_constraints is not None:
            self.hard_constraints = list(hard_constraints)
        return {"ok": True, "worksheet": dataclasses.asdict(self)}


SET_PRIORITIES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "set_priorities",
        "description": (
            "Record or update the user's priorities worksheet. Call this as "
            "soon as the user states or you infer tf_known, top_priority, or "
            "hard_constraints -- even partially. Fields not provided are left "
            "unchanged. tf_known cannot be changed once set."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tf_known": {
                    "type": "boolean",
                    "description": "True if the user knows their plant's transfer function, false if only signal/experiment data is available.",
                },
                "top_priority": {
                    "type": "string",
                    "enum": list(PRIORITY_CATEGORIES.keys()),
                    "description": "The single metric category the user cares about most.",
                },
                "hard_constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Free-text hard constraints, e.g. 'no derivative action'.",
                },
            },
        },
    },
}

FINALIZE_RECOMMENDATION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "finalize_recommendation",
        "description": (
            "Record your final single-method recommendation. method_name MUST "
            "exactly match a stable/available 'name' field already returned by "
            "a benchmark tool call in this conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "method_name": {"type": "string"},
                "rationale": {
                    "type": "string",
                    "description": "1-3 sentences citing specific metric values from the tool result.",
                },
            },
            "required": ["method_name", "rationale"],
        },
    },
}


def make_set_priorities_tool(worksheet: PrioritiesWorksheet):
    """Bind `set_priorities` to a specific worksheet instance."""

    def _tool(tf_known=None, top_priority=None, hard_constraints=None):
        return worksheet.update(
            tf_known=tf_known, top_priority=top_priority,
            hard_constraints=hard_constraints,
        )

    return _tool


def make_finalize_recommendation_tool(known_stable_methods: set):
    """Bind `finalize_recommendation` to a session's live grounding cache.

    `known_stable_methods` is passed by reference (a set) and read at call
    time, so it always reflects whatever benchmark rows have been returned
    so far in the session.
    """

    def _tool(method_name=None, rationale=None):
        if not method_name or method_name not in known_stable_methods:
            valid = sorted(known_stable_methods)
            return {
                "ok": False,
                "error": (
                    f"{method_name!r} was never returned as a stable/available "
                    f"method by a benchmark tool in this conversation. Valid "
                    f"choices: {valid}. Call a benchmark tool first, or pick "
                    "one of the valid choices."
                ),
            }
        return {"ok": True, "method_name": method_name, "rationale": rationale}

    return _tool
