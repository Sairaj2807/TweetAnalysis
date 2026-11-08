import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config.settings import HORIZONS


def plot_actual_vs_predicted(test_df: pd.DataFrame, preds: pd.DataFrame, out_path: str, title: str = "Actual vs Predicted Price") -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test_df["timestamp_utc"], test_df["price"], label="Actual price", linewidth=2, color="black")

    for h in HORIZONS:
        if h == 0:
            continue
        ax.plot(test_df["timestamp_utc"], preds[f"n{h}"], label=f"Predicted n+{h}", linestyle="--", alpha=0.7)

    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.set_title(title)
    ax.legend()
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
