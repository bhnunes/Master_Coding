from __future__ import annotations

from pathlib import Path

import pytest

from helpers.runtime_platform import load_openslide_module, resolve_env_path


def test_resolve_env_path_rejects_windows_path_on_linux() -> None:
    with pytest.raises(ValueError, match="Windows path"):
        resolve_env_path(r"D:\slides\case01.svs", "PROJECTS_BASE_PATH", system_name="Linux")


def test_resolve_env_path_rejects_posix_absolute_path_on_windows() -> None:
    with pytest.raises(ValueError, match="POSIX-style path"):
        resolve_env_path("/data/slides", "PROJECTS_BASE_PATH", system_name="Windows")


def test_resolve_env_path_normalizes_windows_separators() -> None:
    resolved = resolve_env_path(
        r"C:\OpenSlide\bin",
        "OPENSLIDE_PATH",
        system_name="Windows",
    )

    assert resolved == Path("C:/OpenSlide/bin")


def test_resolve_env_path_rejects_control_characters() -> None:
    with pytest.raises(ValueError, match="contains control characters"):
        resolve_env_path(
            "F:\\CHILE_OUTPUT\\PATCHES\x07ccepted_manifest.csv",
            "PACKAGING_ACCEPTED_MANIFEST_PATH",
            system_name="Windows",
        )


def test_load_openslide_module_bypasses_openslide_path_on_linux() -> None:
    imported_modules: list[str] = []
    sentinel = object()

    def fake_import_module(name: str) -> object:
        imported_modules.append(name)
        return sentinel

    loaded_module = load_openslide_module(
        {"OPENSLIDE_PATH": r"D:\OpenSlide\bin"},
        system_name="Linux",
        import_module_fn=fake_import_module,
    )

    assert loaded_module is sentinel
    assert imported_modules == ["openslide"]


def test_load_openslide_module_requires_openslide_path_on_windows() -> None:
    with pytest.raises(ValueError, match="OPENSLIDE_PATH"):
        load_openslide_module({}, system_name="Windows")


def test_load_openslide_module_uses_add_dll_directory_on_windows() -> None:
    events: list[str] = []
    sentinel = object()

    class DummyContext:
        def __enter__(self) -> None:
            events.append("enter")

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            events.append("exit")

    def fake_import_module(name: str) -> object:
        events.append(name)
        return sentinel

    def fake_add_dll_directory(path: str) -> DummyContext:
        events.append(path)
        return DummyContext()

    loaded_module = load_openslide_module(
        {"OPENSLIDE_PATH": r"C:\OpenSlide\bin"},
        system_name="Windows",
        import_module_fn=fake_import_module,
        add_dll_directory_fn=fake_add_dll_directory,
        is_dir_fn=lambda _: True,
    )

    assert loaded_module is sentinel
    assert events == ["C:/OpenSlide/bin", "enter", "openslide", "exit"]
