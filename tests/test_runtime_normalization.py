from __future__ import annotations

import pytest

from helpers.runtime_normalization import format_runtime_normalization_status


def test_format_runtime_normalization_status_reports_method_without_backend() -> None:
    assert (
        format_runtime_normalization_status(
            runtime_normalization_method="macenko",
            runtime_vahadane_backend="fixed_source",
        )
        == "Runtime normalization: MACENKO"
    )


def test_format_runtime_normalization_status_reports_vahadane_backend() -> None:
    assert (
        format_runtime_normalization_status(
            runtime_normalization_method="vahadane",
            runtime_vahadane_backend="torch_staintools_exact",
        )
        == "Runtime normalization: VAHADANE (backend=torch_staintools_exact)"
    )


def test_format_runtime_normalization_status_rejects_invalid_method() -> None:
    with pytest.raises(ValueError, match="RUNTIME_NORMALIZATION_METHOD"):
        format_runtime_normalization_status(
            runtime_normalization_method="invalid",
            runtime_vahadane_backend="fixed_source",
        )
