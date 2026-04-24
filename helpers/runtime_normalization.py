from __future__ import annotations

RUNTIME_NORMALIZATION_METHOD_ENV_VAR = "RUNTIME_NORMALIZATION_METHOD"
VALID_RUNTIME_NORMALIZATION_METHODS = frozenset(
    {"NOT_NORMALIZED", "REINHARD", "RUIFROK", "MACENKO", "VAHADANE"}
)


def parse_runtime_normalization_method(
    value: str | None,
    *,
    variable_name: str = RUNTIME_NORMALIZATION_METHOD_ENV_VAR,
    default: str = "NOT_NORMALIZED",
) -> str:
    """Parse the shared runtime normalization selector used by Phases 7-10."""

    candidate = (value or default).strip().upper()
    if candidate not in VALID_RUNTIME_NORMALIZATION_METHODS:
        choices = ", ".join(sorted(VALID_RUNTIME_NORMALIZATION_METHODS))
        raise ValueError(f"The '{variable_name}' environment variable must be one of: {choices}.")
    return candidate
