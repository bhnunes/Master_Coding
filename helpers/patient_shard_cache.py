from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

_CACHE_PROMOTION_LOCK = threading.Lock()


@dataclass(frozen=True)
class PatientShardCache:
    cache_dir: Path
    size_cap_bytes: int
    _resolved_paths: dict[Path, Path] = field(default_factory=dict, init=False, repr=False)

    def fetch(self, source_path: Path) -> Path:
        resolved_path = self._resolved_paths.get(source_path)
        if resolved_path is not None and (self.size_cap_bytes <= 0 or resolved_path.exists()):
            return resolved_path
        if resolved_path is not None:
            self._resolved_paths.pop(source_path, None)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cached_path = self.cache_dir / self._cache_file_name(source_path)
        if cached_path.exists():
            if self.size_cap_bytes > 0:
                cached_path.touch()
            self._resolved_paths[source_path] = cached_path
            return cached_path

        temp_path = self._reserve_temp_path(cached_path)
        try:
            shutil.copy2(source_path, temp_path)
            self._promote_temp_path(temp_path=temp_path, cached_path=cached_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        if self.size_cap_bytes > 0:
            cached_path.touch()
        self._evict_if_needed(protected_path=cached_path)
        self._resolved_paths[source_path] = cached_path
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

    def _reserve_temp_path(self, cached_path: Path) -> Path:
        file_descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{cached_path.name}.",
            suffix=f".{os.getpid()}.tmp",
            dir=self.cache_dir,
        )
        os.close(file_descriptor)
        return Path(temp_name)

    @staticmethod
    def _promote_temp_path(*, temp_path: Path, cached_path: Path) -> None:
        with _CACHE_PROMOTION_LOCK:
            if cached_path.exists():
                temp_path.unlink(missing_ok=True)
                cached_path.touch()
                return
            try:
                temp_path.replace(cached_path)
            except PermissionError:
                if not cached_path.exists():
                    raise
                temp_path.unlink(missing_ok=True)
                cached_path.touch()
