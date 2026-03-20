from __future__ import annotations

from dotenv import load_dotenv

from helpers.ensemble_optimizer.config import load_ensemble_optimizer_config
from helpers.ensemble_optimizer.pipeline import run_ensemble_optimizer_pipeline


def main() -> None:
    load_dotenv(override=True)
    config = load_ensemble_optimizer_config()
    outputs = run_ensemble_optimizer_pipeline(config)
    print(f"Ensemble recipe saved to: {outputs.recipe_path}")
    print(f"Run config saved to: {outputs.run_config_path}")


if __name__ == "__main__":
    main()
