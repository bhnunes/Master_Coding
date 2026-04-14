from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PatientShardCache:
    cache_dir: Path
    size_cap_bytes: int

    def fetch(self, source_path: Path) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cached_path = self.cache_dir / self._cache_file_name(source_path)
        if cached_path.exists():
            cached_path.touch()
            return cached_path

        temp_path = cached_path.with_name(f".{cached_path.name}.tmp")
        if temp_path.exists():
            temp_path.unlink()
        shutil.copy2(source_path, temp_path)
        temp_path.replace(cached_path)
        cached_path.touch()
        self._evict_if_needed(protected_path=cached_path)
        return cached_path

    def _evict_if_needed(self, *, protected_path: Path) -> None:
        if self.size_cap_bytes <= 0:
            return
        cached_files = sorted(
            (path for path in self.cache_dir.glob("*.h5") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
        )
        total_bytes = sum(path.stat().st_size for path in cached_files)
        for path in cached_files:
            if total_bytes <= self.size_cap_bytes:
                break
            if path == protected_path:
                continue
            size = path.stat().st_size
            path.unlink()
            total_bytes -= size

    @staticmethod
    def _cache_file_name(source_path: Path) -> str:
        digest = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:12]
        return f"{source_path.stem}-{digest}{source_path.suffix}"
