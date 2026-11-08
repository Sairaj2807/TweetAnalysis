import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_sentiment_heatmap(df: pd.DataFrame, out_path: str, title: str = "Sentiment Heatmap") -> None:
    """Plots a day-vs-hour heatmap of average sentiment, colored red (negative) to green (positive)."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["timestamp_utc"]).dt.date
    d["hour"] = pd.to_datetime(d["timestamp_utc"]).dt.hour

    pivot = d.pivot_table(index="date", columns="hour", values="sentiment_mean", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(14, max(4, 0.3 * len(pivot))))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(d) for d in pivot.index])
    ax.set_xlabel("Hour of day (UTC)")
    ax.set_ylabel("Date")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Mean sentiment")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
