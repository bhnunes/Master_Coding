from __future__ import annotations

from dotenv import load_dotenv

from helpers.ensemble_inference.config import load_ensemble_inference_config
from helpers.ensemble_inference.pipeline import run_ensemble_inference_pipeline


def main() -> None:
    load_dotenv(override=True)
    config = load_ensemble_inference_config()
    outputs = run_ensemble_inference_pipeline(config)
    print(f"Inference output directory: {outputs.output_dir}")
    print(f"Run config saved to: {outputs.run_config_path}")
    print(f"Metrics JSON saved to: {outputs.metrics_json_path}")
    print(f"Confusion matrix saved to: {outputs.confusion_matrix_path}")
    if outputs.csv_report_path is not None:
        print(f"CSV report saved to: {outputs.csv_report_path}")
    if outputs.pdf_report_path is not None:
        print(f"PDF report saved to: {outputs.pdf_report_path}")


if __name__ == "__main__":
    main()
