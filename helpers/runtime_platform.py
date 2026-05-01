from __future__ import annotations

import os
import platform
from collections.abc import MutableMapping
from contextlib import AbstractContextManager, nullcontext
from importlib import import_module
from pathlib import Path
from typing import Any, cast

CONTROL_CHARACTER_MAX_ORDINAL = 31
WINDOWS_DRIVE_PREFIX_LENGTH = 3
COLAB_INLINE_MATPLOTLIB_BACKEND = "module://matplotlib_inline.backend_inline"
HEADLESS_MATPLOTLIB_BACKEND = "Agg"


def get_runtime_os(system_name: str | None = None) -> str:
    """Return the normalized runtime operating system name."""

    return system_name or platform.system()


def ensure_headless_matplotlib_backend(
    env: MutableMapping[str, str] | os._Environ[str] | None = None,
) -> None:
    """Use a supported non-interactive backend when Colab exports its inline backend."""

    values = env if env is not None else os.environ
    if values.get("MPLBACKEND") == COLAB_INLINE_MATPLOTLIB_BACKEND:
        values["MPLBACKEND"] = HEADLESS_MATPLOTLIB_BACKEND


def load_headless_matplotlib_pyplot() -> Any:
    """Load matplotlib.pyplot with a backend that works in headless pipeline runs."""

    ensure_headless_matplotlib_backend()

    import matplotlib

    matplotlib.use(HEADLESS_MATPLOTLIB_BACKEND, force=True)
    from matplotlib import pyplot

    return pyplot


def resolve_env_path(
    value: str | None,
    variable_name: str,
    *,
    system_name: str | None = None,
    required: bool = False,
    require_native: bool = True,
) -> Path | None:
    """Sanitize a path-like environment variable for the active runtime."""

    if value is None:
        if required:
            raise ValueError(f"The '{variable_name}' environment variable is required.")
        return None

    stripped = value.strip()
    if not stripped:
        if required:
            raise ValueError(f"The '{variable_name}' environment variable is required.")
        return None

    if any(ord(character) <= CONTROL_CHARACTER_MAX_ORDINAL for character in stripped):
        raise ValueError(
            f"The '{variable_name}' environment variable contains control characters. "
            "This often happens when a Windows path in .env uses backslashes that are "
            "interpreted as escapes. Prefer forward slashes, for example "
            "'F:/CHILE_OUTPUT/PATCHES/accepted_manifest.csv'."
        )

    runtime_os = get_runtime_os(system_name)
    windows_style = _is_windows_style_path(stripped)
    posix_absolute = stripped.startswith("/")

    if require_native and runtime_os != "Windows" and windows_style:
        raise ValueError(
            f"Windows path '{stripped}' is not valid for '{variable_name}' on {runtime_os}."
        )

    if require_native and runtime_os == "Windows" and posix_absolute:
        raise ValueError(
            f"POSIX-style path '{stripped}' is not valid for '{variable_name}' on Windows."
        )

    normalized = stripped.replace("\\", "/") if windows_style else stripped
    return Path(normalized).expanduser()


def _path_for_runtime_api(path: Path, *, runtime_os: str) -> str:
    if runtime_os == "Windows":
        return path.as_posix()
    return str(path)


def get_openslide_dll_context(
    env: dict[str, str | None] | os._Environ[str] | None = None,
    *,
    system_name: str | None = None,
    add_dll_directory_fn: Any | None = None,
    is_dir_fn: Any | None = None,
) -> AbstractContextManager[None]:
    """Return a context manager that configures OpenSlide DLL lookup on Windows."""

    runtime_os = get_runtime_os(system_name)
    if runtime_os != "Windows":
        return nullcontext()

    values = env if env is not None else os.environ
    openslide_path = resolve_env_path(
        values.get("OPENSLIDE_PATH"),
        "OPENSLIDE_PATH",
        system_name=runtime_os,
        required=True,
    )
    assert openslide_path is not None

    add_dll_directory = add_dll_directory_fn
    if add_dll_directory is None:
        add_dll_directory = getattr(os, "add_dll_directory", None)
    if not callable(add_dll_directory):
        raise RuntimeError("OpenSlide DLL loading requires os.add_dll_directory on Windows.")

    is_dir = is_dir_fn or os.path.isdir
    openslide_path_str = _path_for_runtime_api(openslide_path, runtime_os=runtime_os)
    if not is_dir(openslide_path_str):
        raise ValueError(
            "OPENSLIDE_PATH must point to the OpenSlide 'bin' directory on Windows. "
            f"Got: {openslide_path_str}"
        )

    return cast(AbstractContextManager[None], add_dll_directory(openslide_path_str))


def load_openslide_module(
    env: dict[str, str | None] | os._Environ[str] | None = None,
    *,
    system_name: str | None = None,
    import_module_fn: Any | None = None,
    add_dll_directory_fn: Any | None = None,
    is_dir_fn: Any | None = None,
) -> Any:
    """Import and return the OpenSlide Python module with OS-aware setup."""

    runtime_os = get_runtime_os(system_name)
    importer = import_module_fn or import_module
    if runtime_os != "Windows":
        return importer("openslide")

    with get_openslide_dll_context(
        env,
        system_name=runtime_os,
        add_dll_directory_fn=add_dll_directory_fn,
        is_dir_fn=is_dir_fn,
    ):
        return importer("openslide")


def _is_windows_style_path(value: str) -> bool:
    return (
        len(value) >= WINDOWS_DRIVE_PREFIX_LENGTH
        and value[1] == ":"
        and value[0].isalpha()
        and value[2] in {"\\", "/"}
    ) or value.startswith("\\\\")
