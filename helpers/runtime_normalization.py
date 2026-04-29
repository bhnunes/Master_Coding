from __future__ import annotations

RUNTIME_NORMALIZATION_METHOD_ENV_VAR = "RUNTIME_NORMALIZATION_METHOD"
RUNTIME_VAHADANE_BACKEND_ENV_VAR = "RUNTIME_VAHADANE_BACKEND"
VALID_RUNTIME_NORMALIZATION_METHODS = frozenset(
    {"NOT_NORMALIZED", "REINHARD", "RUIFROK", "MACENKO", "VAHADANE"}
)
VALID_RUNTIME_VAHADANE_BACKENDS = frozenset({"fixed_source", "torch_staintools_exact"})


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


def parse_runtime_vahadane_backend(
    value: str | None,
    *,
    variable_name: str = RUNTIME_VAHADANE_BACKEND_ENV_VAR,
    default: str = "fixed_source",
) -> str:
    """Parse the VAHADANE runtime backend selector shared by Phases 7-10."""

    candidate = (value or default).strip().lower()
    if candidate not in VALID_RUNTIME_VAHADANE_BACKENDS:
        choices = ", ".join(sorted(VALID_RUNTIME_VAHADANE_BACKENDS))
        raise ValueError(f"The '{variable_name}' environment variable must be one of: {choices}.")
    return candidate


def runtime_vahadane_backend_for_provenance(
    *,
    runtime_normalization_method: str,
    runtime_vahadane_backend: str,
) -> str | None:
    """Return the active VAHADANE backend value for lineage payloads."""

    if runtime_normalization_method.strip().lower() not in {"vahadane"}:
        return None
    return parse_runtime_vahadane_backend(runtime_vahadane_backend)
