import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_sentiment_vs_price(df: pd.DataFrame, out_path: str, title: str = "Sentiment vs Price") -> None:
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(df["timestamp_utc"], df["price"], color="tab:blue", label="Price")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Price", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(df["timestamp_utc"], df["sentiment_mean"], color="tab:green", alpha=0.6, label="X sentiment")
    if "news_sentiment" in df.columns:
        ax2.plot(df["timestamp_utc"], df["news_sentiment"], color="tab:orange", alpha=0.5, label="News sentiment")
    ax2.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax2.set_ylabel("Sentiment score", color="tab:green")

    fig.suptitle(title)
    fig.legend(loc="upper left")
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
