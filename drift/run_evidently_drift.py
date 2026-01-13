from evidently import Report
from evidently.presets import DataDriftPreset

from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/drift")
REPORT_DIR = Path("reports/drift")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    reference_df = pd.read_parquet(DATA_DIR / "reference.parquet")
    current_df = pd.read_parquet(DATA_DIR / "current.parquet")

    report = Report(
        metrics=[DataDriftPreset()]
    )

    my_eval = report.run(
        reference_data=reference_df,
        current_data=current_df
    )
    
    html_output_path = REPORT_DIR / "data_drift_report.html"
    
    my_eval.save_html(str(html_output_path))
    
    print(f"Drift report saved at: {html_output_path.resolve()}")

if __name__ == "__main__":
    main()