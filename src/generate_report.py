"""
Generates the final Markdown report.
Combines visualizations and their corresponding tables using first-person language.
"""
import pandas as pd
from pathlib import Path

def build_report():
    """Reads visualization images and table CSVs to compile FINAL_REPORT.md."""
    report_path = "FINAL_REPORT.md"
    viz_dir = Path("data/visualizations")
    table_dir = Path("data/tables")

    with open(report_path, "w") as f:
        f.write("# My Medical Image Classification Report\n\n")
        f.write("This report details my findings, model performance metrics, and feature analysis.\n\n")

        # Feature Impact Section
        f.write("## 1. Feature Impact\n")
        f.write("![Feature Impact](data/visualizations/feature_impact.png)\n\n")
        impact_csv = table_dir / "feature_impact.csv"
        if impact_csv.exists():
            f.write(pd.read_csv(impact_csv).to_markdown(index=False) + "\n\n")

        # AUC Curve Section
        f.write("## 2. AUC / ROC Curve\n")
        f.write("![AUC Curve](data/visualizations/auc_roc.png)\n\n")
        auc_csv = table_dir / "auc_roc.csv"
        if auc_csv.exists():
            f.write(pd.read_csv(auc_csv).to_markdown(index=False) + "\n\n")

        # Gallery for remaining visuals
        f.write("## 3. Additional Visualizations\n")
        for img in sorted(viz_dir.glob("*.png")):
            if img.name not in ["feature_impact.png", "auc_roc.png"]:
                f.write(f"### {img.stem.replace('_', ' ').title()}\n")
                f.write(f"![{img.name}](data/visualizations/{img.name})\n\n")
                
                # Check for a matching table and append it if it exists
                possible_table = table_dir / f"{img.stem}.csv"
                if possible_table.exists():
                    f.write(pd.read_csv(possible_table).to_markdown(index=False) + "\n\n")

    print(f"--- SUCCESS: Report saved to {report_path} ---")

if __name__ == "__main__":
    build_report()
