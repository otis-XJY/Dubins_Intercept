from __future__ import annotations

from typing import Optional


SCHEME_KEY_TO_NAME = {
    "A": "Concatenative Query Network",
    "B": "Gated Query Network",
    "C": "Point-Wise Scoring Network",
}

_SCHEME_ALIAS_TO_KEY = {
    "a": "A",
    "concatenative query network": "A",
    "concatenative_query_network": "A",
    "cqn": "A",
    "concat": "A",
    "b": "B",
    "gated query network": "B",
    "gated_query_network": "B",
    "gqn": "B",
    "soft-gating": "B",
    "soft_gating": "B",
    "c": "C",
    "point-wise scoring network": "C",
    "point_wise_scoring_network": "C",
    "pointwise scoring network": "C",
    "pwsn": "C",
}


def resolve_design_mode(design_mode: Optional[str] = None, use_soft_gating: Optional[bool] = None) -> str:
    """Resolve model design alias to canonical key: A/B/C."""
    if design_mode is None:
        if use_soft_gating is None:
            return "A"
        return "B" if bool(use_soft_gating) else "A"

    normalized = str(design_mode).strip().lower()
    if normalized in _SCHEME_ALIAS_TO_KEY:
        return _SCHEME_ALIAS_TO_KEY[normalized]

    valid_names = ", ".join(SCHEME_KEY_TO_NAME.values())
    raise ValueError(f"Unknown design_mode='{design_mode}'. Use one of A/B/C or: {valid_names}.")


def design_mode_to_name(design_mode: str) -> str:
    return SCHEME_KEY_TO_NAME[resolve_design_mode(design_mode)]
