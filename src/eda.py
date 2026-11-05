from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import REPORTS_DIR


def generate_eda_report(df):
    """Create a set of standard exploratory charts for the churn dataset."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    churn_counts = df["churn"].value_counts().to_dict()
    figure_path = REPORTS_DIR / "churn_distribution.png"
    plt.figure(figsize=(6, 5))
    sns.countplot(data=df, x="churn")
    plt.title("Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()

    monthly = REPORTS_DIR / "monthly_charges_boxplot.png"
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="churn", y="monthlycharges")
    plt.title("Monthly Charges by Churn Status")
    plt.xlabel("Churn")
    plt.ylabel("Monthly Charges")
    plt.tight_layout()
    plt.savefig(monthly, dpi=200)
    plt.close()

    tenure_path = REPORTS_DIR / "tenure_distribution.png"
    plt.figure(figsize=(7, 5))
    sns.histplot(data=df, x="tenure", hue="churn", kde=True, element="step")
    plt.title("Tenure Distribution by Churn Status")
    plt.tight_layout()
    plt.savefig(tenure_path, dpi=200)
    plt.close()

    return {
        "churn_counts": churn_counts,
        "figures": {
            "churn_distribution": str(figure_path),
            "monthly_charges": str(monthly),
            "tenure_distribution": str(tenure_path),
        },
    }
