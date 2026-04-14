from __future__ import annotations

from pathlib import Path

from helpers.patient_shard_cache import PatientShardCache


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
