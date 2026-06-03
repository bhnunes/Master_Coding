from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

SPLITS = ("TRAIN", "VALIDATION", "TEST")
LABEL_DIR = {0: "NOT_CANCER", 1: "CANCER"}
MASK_DIR = {0: "NOT_CANCER_MASK", 1: "CANCER_MASK"}
EXPECTED_MASK_VALUES = {0, 255}
STATUS_ORDER = {"FAIL": 3, "WARN": 2, "PASS": 1, "N/A": 0}


@dataclass(frozen=True)
class CheckResult:
    status: str
    details: str
    stats: Mapping[str, object] | None = None


@dataclass(frozen=True)
class SanityReport:
    base_dir: str
    dataset_checks: dict[str, CheckResult]
    split_checks: dict[str, dict[str, CheckResult]] = field(default_factory=dict)
    verdict: str = "SPLITS REJECTED"


def worst_status(statuses: list[str]) -> str:
    if not statuses:
        return "N/A"
    return max(statuses, key=lambda status: STATUS_ORDER.get(status, 0))
