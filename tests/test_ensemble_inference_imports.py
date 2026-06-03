from __future__ import annotations

import os
import subprocess
import sys

from helpers.runtime_platform import COLAB_INLINE_MATPLOTLIB_BACKEND


def test_ensemble_inference_config_imports_with_colab_inline_backend() -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = COLAB_INLINE_MATPLOTLIB_BACKEND

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from helpers.ensemble_inference.config "
                "import load_ensemble_inference_config; "
                "print(load_ensemble_inference_config.__name__)"
            ),
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert completed.stdout.strip() == "load_ensemble_inference_config"
