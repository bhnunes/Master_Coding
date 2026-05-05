from __future__ import annotations

import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from helpers.patient_shard_cache import PatientShardCache

_CONCURRENT_FETCH_COUNT = 2


def _write_bytes(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_patient_shard_cache_reuses_cached_copy(tmp_path: Path) -> None:
    source_path = tmp_path / "drive" / "1.h5"
    _write_bytes(source_path, 32)
    cache = PatientShardCache(cache_dir=tmp_path / "cache", size_cap_bytes=1024)

    first_cached_path = cache.fetch(source_path)
    original_bytes = first_cached_path.read_bytes()
    source_path.write_bytes(b"y" * 32)

    second_cached_path = cache.fetch(source_path)

    assert second_cached_path == first_cached_path
    assert second_cached_path.read_bytes() == original_bytes


def test_patient_shard_cache_unlimited_cache_memoizes_hot_fetches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_path = tmp_path / "drive" / "1.h5"
    _write_bytes(source_path, 32)
    cache = PatientShardCache(cache_dir=tmp_path / "cache", size_cap_bytes=0)

    first_cached_path = cache.fetch(source_path)

    def fail_touch(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Unlimited cache hot fetch should not touch the filesystem")

    monkeypatch.setattr("helpers.patient_shard_cache.Path.touch", fail_touch)
    second_cached_path = cache.fetch(source_path)

    assert second_cached_path == first_cached_path


def test_patient_shard_cache_evicts_lru_when_size_cap_exceeded(tmp_path: Path) -> None:
    first_source = tmp_path / "drive" / "1.h5"
    second_source = tmp_path / "drive" / "2.h5"
    _write_bytes(first_source, 80)
    _write_bytes(second_source, 80)
    cache = PatientShardCache(cache_dir=tmp_path / "cache", size_cap_bytes=120)

    first_cached_path = cache.fetch(first_source)
    second_cached_path = cache.fetch(second_source)

    assert not first_cached_path.exists()
    assert second_cached_path.exists()

    refreshed_first_path = cache.fetch(first_source)

    assert refreshed_first_path.exists()


def test_patient_shard_cache_uses_unique_temp_paths_for_concurrent_fetches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_path = tmp_path / "drive" / "1.h5"
    _write_bytes(source_path, 32)
    cache = PatientShardCache(cache_dir=tmp_path / "cache", size_cap_bytes=1024)
    barrier = threading.Barrier(_CONCURRENT_FETCH_COUNT)
    temp_paths: list[Path] = []
    temp_paths_lock = threading.Lock()

    def fake_copy2(src: Path, dst: Path) -> Path:
        with temp_paths_lock:
            temp_paths.append(Path(dst))
        barrier.wait(timeout=5)
        shutil.copyfile(src, dst)
        return dst

    monkeypatch.setattr("helpers.patient_shard_cache.shutil.copy2", fake_copy2)

    with ThreadPoolExecutor(max_workers=_CONCURRENT_FETCH_COUNT) as executor:
        results = list(
            executor.map(cache.fetch, [source_path for _ in range(_CONCURRENT_FETCH_COUNT)])
        )

    assert results[0] == results[1]
    assert results[0].exists()
    assert len(set(temp_paths)) == _CONCURRENT_FETCH_COUNT
    assert all(not path.exists() for path in temp_paths)
