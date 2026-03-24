import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_reports():
    base_dir = Path.cwd()
    data_path = base_dir / "data" / "submissions" / "prediction_test_data.csv"
    viz_dir = base_dir / "data" / "visualizations"
    table_dir = base_dir / "data" / "tables"
    
    viz_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"Error: Could not find {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # 3. Create Summary Table
    summary = df['label'].value_counts().reset_index()
    summary.columns = ['Class', 'Count']
    summary['Percentage'] = (summary['Count'] / len(df) * 100).round(2)
    summary['Class_Name'] = summary['Class'].map({0: 'Normal (0)', 1: 'Biomarker Positive (1)'})
    
    table_output = table_dir / "dataset_summary.csv"
    summary.to_csv(table_output, index=False)
    
    # 4. Create Visualization using standard Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#4c72b0', '#55a868'] # Standard blue/green
    
    bars = ax.bar(summary['Class_Name'], summary['Count'], color=colors, edgecolor='black', alpha=0.8)
    
    ax.set_title(f'Distribution of Indexed Images (Total: {len(df)})', fontsize=15, pad=20)
    ax.set_xlabel('Diagnostic Category', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plot_output = viz_dir / "class_distribution.png"
    plt.tight_layout()
    plt.savefig(plot_output)
    
    print("\n" + "="*40)
    print(f"--- SUCCESS: Reports Generated ---")
    print(f"Table: {table_output}")
    print(f"Plot:  {plot_output}")
    print("="*40 + "\n")
    print(summary[['Class_Name', 'Count', 'Percentage']])

if __name__ == "__main__":
    generate_reports()
