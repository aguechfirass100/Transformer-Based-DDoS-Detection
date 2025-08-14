import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def feature_heatmap(df: pd.DataFrame, label_column: str = None, title: str = "Feature Correlation Heatmap"):

    if label_column and label_column in df.columns:
        df = df.drop(columns=[label_column])

    corr_matrix = df.corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        annot=False,
        fmt=".2f",
        cbar=True,
        square=True,
        linewidths=0.5
    )

    plt.title(title, fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.show()
