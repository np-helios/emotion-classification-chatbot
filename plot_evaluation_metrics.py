from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_DIR / "report_artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"


def style_axes(ax, title: str, ylabel: str = "Score") -> None:
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_bar_labels(ax) -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)


def plot_grouped_bars(df: pd.DataFrame, columns: list[str], labels: list[str], title: str, filename: str, ylim=(0, 1.05)) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(df))
    width = 0.18 if len(columns) >= 4 else 0.22
    offsets = [((idx - (len(columns) - 1) / 2) * width) for idx in range(len(columns))]
    colors = ["#2f80ed", "#27ae60", "#9b51e0", "#f2994a", "#eb5757"]

    for idx, column in enumerate(columns):
        ax.bar(
            [pos + offsets[idx] for pos in x],
            df[column],
            width=width,
            label=labels[idx],
            color=colors[idx % len(colors)],
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["method"], rotation=10, ha="right")
    ax.set_ylim(*ylim)
    style_axes(ax, title)
    ax.legend(frameon=False)
    add_bar_labels(ax)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_heatmap(df: pd.DataFrame, metric_columns: list[str], title: str, filename: str) -> None:
    values = df[metric_columns].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    image = ax.imshow(values, cmap="Blues", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(metric_columns)))
    ax.set_xticklabels(metric_columns, rotation=30, ha="right")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["method"])
    ax.set_title(title, fontsize=14, weight="bold")

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center", color="black", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_validation_trials(history_df: pd.DataFrame, methods: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {
        "Linear SVM + TF-IDF": "#2f80ed",
        "Logistic Regression + TF-IDF": "#27ae60",
        "Naive Bayes + BoW": "#9b51e0",
    }

    for method in methods:
        subset = history_df[history_df["method"] == method]
        ax.plot(
            subset["trial"],
            subset["val_f1"],
            marker="o",
            linewidth=2,
            label=method,
            color=colors.get(method, None),
        )
        for _, row in subset.iterrows():
            ax.text(row["trial"], row["val_f1"] + 0.005, f"{row['val_f1']:.3f}", fontsize=8, ha="center")

    style_axes(ax, "Validation Macro F1 Across Hyperparameter Trials", ylabel="Validation Macro F1")
    ax.set_xlabel("Trial")
    ax.set_xticks(sorted(history_df["trial"].unique()))
    ax.set_ylim(0.68, 0.89)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "validation_trials.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.read_csv(ARTIFACTS_DIR / "results_table.csv")
    history_df = pd.read_csv(ARTIFACTS_DIR / "training_history.csv")

    plot_grouped_bars(
        results_df,
        ["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        ["Accuracy", "Precision", "Recall", "F1-score"],
        "Core Performance Metrics Comparison",
        "core_metrics_comparison.png",
    )

    plot_grouped_bars(
        results_df,
        ["sensitivity", "specificity", "fpr", "fnr"],
        ["Sensitivity", "Specificity", "FPR", "FNR"],
        "Sensitivity, Specificity, FPR and FNR",
        "error_metrics_comparison.png",
    )

    plot_grouped_bars(
        results_df,
        ["npv", "fdr", "mcc"],
        ["NPV", "FDR", "MCC"],
        "NPV, FDR and MCC Comparison",
        "npv_fdr_mcc_comparison.png",
    )

    plot_metric_heatmap(
        results_df,
        ["accuracy", "precision_macro", "recall_macro", "f1_macro", "mcc"],
        "Model Performance Heatmap",
        "performance_heatmap.png",
    )

    plot_validation_trials(
        history_df,
        ["Linear SVM + TF-IDF", "Logistic Regression + TF-IDF", "Naive Bayes + BoW"],
    )

    print(f"Saved plots to: {PLOTS_DIR}")
    for plot_file in sorted(PLOTS_DIR.glob("*.png")):
        print(plot_file.name)


if __name__ == "__main__":
    main()
