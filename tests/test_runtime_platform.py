from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

from helpers import runtime_platform
from helpers.runtime_platform import (
    ensure_headless_matplotlib_backend,
    get_openslide_dll_context,
    load_headless_matplotlib_pyplot,
    load_openslide_module,
    resolve_env_path,
    suppress_native_stderr,
)


def test_resolve_env_path_returns_none_for_missing_optional_value() -> None:
    assert resolve_env_path(None, "OPTIONAL_PATH") is None


def test_resolve_env_path_signature_defaults_are_stable() -> None:
    signature = inspect.signature(runtime_platform.resolve_env_path)

    assert signature.parameters["required"].default is False
    assert signature.parameters["require_native"].default is True


def test_resolve_env_path_namespace_uses_default_optional_behavior() -> None:
    assert runtime_platform.resolve_env_path(None, "OPTIONAL_PATH") is None


def test_get_runtime_os_prefers_explicit_value_and_platform_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("platform.system", lambda: "Darwin")

    assert runtime_platform.get_runtime_os("Linux") == "Linux"
    assert runtime_platform.get_runtime_os() == "Darwin"


def test_ensure_headless_matplotlib_backend_replaces_colab_inline_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MPLBACKEND", runtime_platform.COLAB_INLINE_MATPLOTLIB_BACKEND)

    ensure_headless_matplotlib_backend()

    assert os.environ["MPLBACKEND"] == runtime_platform.HEADLESS_MATPLOTLIB_BACKEND


def test_ensure_headless_matplotlib_backend_keeps_supported_backend() -> None:
    env = {"MPLBACKEND": "svg"}

    ensure_headless_matplotlib_backend(env)

    assert env["MPLBACKEND"] == "svg"


def test_suppress_native_stderr_redirects_fd_two_temporarily(
    capfd: pytest.CaptureFixture[str],
) -> None:
    os.write(2, b"visible before\n")
    assert "visible before" in capfd.readouterr().err

    with suppress_native_stderr():
        os.write(2, b"hidden native warning\n")
    assert "hidden native warning" not in capfd.readouterr().err

    os.write(2, b"visible after\n")
    assert "visible after" in capfd.readouterr().err


def test_load_headless_matplotlib_pyplot_replaces_colab_inline_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MPLBACKEND", runtime_platform.COLAB_INLINE_MATPLOTLIB_BACKEND)

    pyplot = load_headless_matplotlib_pyplot()

    assert os.environ["MPLBACKEND"] == runtime_platform.HEADLESS_MATPLOTLIB_BACKEND
    assert pyplot.get_backend() == runtime_platform.HEADLESS_MATPLOTLIB_BACKEND


def test_resolve_env_path_does_not_treat_non_alpha_drive_prefix_as_windows_path() -> None:
    resolved = resolve_env_path("1:/slides", "PROJECTS_BASE_PATH", system_name="Linux")

    assert resolved == Path("1:/slides")


def test_is_windows_style_path_detects_drive_and_unc_variants() -> None:
    assert runtime_platform._is_windows_style_path(r"D:\slides\case01.svs") is True
    assert runtime_platform._is_windows_style_path("C:/") is True
    assert runtime_platform._is_windows_style_path(r"\\server\share") is True
    assert runtime_platform._is_windows_style_path("1:/slides") is False
    assert runtime_platform._is_windows_style_path("/data/slides") is False


def test_resolve_env_path_rejects_windows_path_on_linux() -> None:
    with pytest.raises(ValueError, match="Windows path"):
        resolve_env_path(r"D:\slides\case01.svs", "PROJECTS_BASE_PATH", system_name="Linux")


def test_resolve_env_path_namespace_keeps_native_checks_enabled_by_default() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"^Windows path 'D:\\slides\\case01\.svs' is not valid for "
            r"'PROJECTS_BASE_PATH' on Linux\.$"
        ),
    ):
        runtime_platform.resolve_env_path(
            r"D:\slides\case01.svs",
            "PROJECTS_BASE_PATH",
            system_name="Linux",
        )


def test_resolve_env_path_rejects_windows_drive_root_on_linux() -> None:
    with pytest.raises(ValueError, match="Windows path"):
        resolve_env_path("C:/", "OPENSLIDE_PATH", system_name="Linux")


def test_resolve_env_path_rejects_posix_absolute_path_on_windows() -> None:
    with pytest.raises(ValueError, match="POSIX-style path"):
        resolve_env_path("/data/slides", "PROJECTS_BASE_PATH", system_name="Windows")


def test_resolve_env_path_namespace_reports_exact_posix_error_on_windows() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"^POSIX-style path '/data/slides' is not valid for "
            r"'PROJECTS_BASE_PATH' on Windows\.$"
        ),
    ):
        runtime_platform.resolve_env_path(
            "/data/slides",
            "PROJECTS_BASE_PATH",
            system_name="Windows",
        )


def test_resolve_env_path_normalizes_windows_separators() -> None:
    resolved = resolve_env_path(
        r"C:\OpenSlide\bin",
        "OPENSLIDE_PATH",
        system_name="Windows",
    )

    assert resolved == Path("C:/OpenSlide/bin")


def test_resolve_env_path_accepts_windows_drive_root_path() -> None:
    resolved = resolve_env_path("C:/", "OPENSLIDE_PATH", system_name="Windows")

    assert resolved == Path("C:/")


def test_resolve_env_path_accepts_unc_path_on_windows() -> None:
    resolved = resolve_env_path(r"\\server\share", "OPENSLIDE_PATH", system_name="Windows")

    assert resolved == Path("//server/share")


def test_resolve_env_path_rejects_control_characters() -> None:
    with pytest.raises(ValueError, match="contains control characters"):
        resolve_env_path(
            "F:\\CHILE_OUTPUT\\PATCHES\x07ccepted_manifest.csv",
            "PACKAGING_ACCEPTED_MANIFEST_PATH",
            system_name="Windows",
        )


def test_resolve_env_path_namespace_rejects_unit_separator_with_exact_message() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"^The 'PACKAGING_ACCEPTED_MANIFEST_PATH' environment variable "
            r"contains control characters\..*accepted_manifest\.csv'\.$"
        ),
    ):
        runtime_platform.resolve_env_path(
            "F:\\CHILE_OUTPUT\\PATCHES\x1fccepted_manifest.csv",
            "PACKAGING_ACCEPTED_MANIFEST_PATH",
            system_name="Windows",
        )


def test_resolve_env_path_namespace_normalizes_windows_separators_exactly() -> None:
    assert runtime_platform.resolve_env_path(
        r"C:\OpenSlide\bin",
        "OPENSLIDE_PATH",
        system_name="Windows",
    ) == Path("C:/OpenSlide/bin")


def test_runtime_platform_namespace_smoke_path() -> None:
    imported_modules: list[str] = []

    def fake_import_module(name: str) -> object:
        imported_modules.append(name)
        return object()

    assert runtime_platform.get_runtime_os("Linux") == "Linux"
    assert (
        runtime_platform.resolve_env_path(
            "~/slides",
            "PROJECTS_BASE_PATH",
        )
        == Path("~/slides").expanduser()
    )
    assert (
        runtime_platform.load_openslide_module(
            system_name="Linux",
            import_module_fn=fake_import_module,
        )
        is not None
    )
    assert imported_modules == ["openslide"]


def test_get_openslide_dll_context_namespace_validates_required_path_message() -> None:
    with pytest.raises(
        ValueError,
        match=r"^The 'OPENSLIDE_PATH' environment variable is required\.$",
    ):
        runtime_platform.get_openslide_dll_context(
            {},
            system_name="Windows",
            add_dll_directory_fn=lambda _: object(),
            is_dir_fn=lambda _: True,
        )


def test_get_openslide_dll_context_namespace_rejects_non_directory() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"^OPENSLIDE_PATH must point to the OpenSlide 'bin' directory on "
            r"Windows\. Got: C:/bad$"
        ),
    ):
        runtime_platform.get_openslide_dll_context(
            {"OPENSLIDE_PATH": "C:/bad"},
            system_name="Windows",
            add_dll_directory_fn=lambda _: object(),
            is_dir_fn=lambda _: False,
        )


def test_get_openslide_dll_context_namespace_checks_resolved_string_path() -> None:
    checked_paths: list[str | None] = []

    class DummyContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    def fake_is_dir(path: str | None) -> bool:
        checked_paths.append(path)
        return path == "C:/OpenSlide/bin"

    context = runtime_platform.get_openslide_dll_context(
        {"OPENSLIDE_PATH": r"C:\OpenSlide\bin"},
        system_name="Windows",
        add_dll_directory_fn=lambda _: DummyContext(),
        is_dir_fn=fake_is_dir,
    )

    assert checked_paths == ["C:/OpenSlide/bin"]
    assert isinstance(context, DummyContext)


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
    with pytest.raises(
        ValueError,
        match=r"^The 'OPENSLIDE_PATH' environment variable is required\.$",
    ):
        load_openslide_module({}, system_name="Windows")


def test_get_openslide_dll_context_checks_the_resolved_directory_path() -> None:
    checked_paths: list[str] = []

    class DummyContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    def fake_is_dir(path: str) -> bool:
        checked_paths.append(path)
        return path == "C:/OpenSlide/bin"

    context = get_openslide_dll_context(
        {"OPENSLIDE_PATH": r"C:\OpenSlide\bin"},
        system_name="Windows",
        add_dll_directory_fn=lambda path: DummyContext(),
        is_dir_fn=fake_is_dir,
    )

    assert checked_paths == ["C:/OpenSlide/bin"]
    assert isinstance(context, DummyContext)


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


def test_load_openslide_module_namespace_windows_path_uses_runtime_os() -> None:
    events: list[str] = []

    class DummyContext:
        def __enter__(self) -> None:
            events.append("enter")

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            events.append("exit")

    def fake_import_module(name: str) -> str:
        events.append(name)
        return name

    def fake_add_dll_directory(path: str) -> DummyContext:
        events.append(path)
        return DummyContext()

    loaded_module = runtime_platform.load_openslide_module(
        {"OPENSLIDE_PATH": r"C:\OpenSlide\bin"},
        system_name="Windows",
        import_module_fn=fake_import_module,
        add_dll_directory_fn=fake_add_dll_directory,
        is_dir_fn=lambda _: True,
    )

    assert loaded_module == "openslide"
    assert events == ["C:/OpenSlide/bin", "enter", "openslide", "exit"]
