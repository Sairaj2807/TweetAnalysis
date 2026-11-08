import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def render_prediction_matrix_table(matrix: pd.DataFrame, out_path: str, asset: str) -> None:
    """Renders the six-horizon prediction matrix as an image table for the report."""
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(matrix) + 1))
    ax.axis("off")

    display = matrix.copy()
    display["predicted_price"] = display["predicted_price"].map(lambda x: f"{x:,.2f}")
    display["change_pct"] = display["change_pct"].map(lambda x: f"{x:+.2f}%")

    table = ax.table(
        cellText=display.values,
        colLabels=["Horizon", "Predicted Price", "Change", "Direction"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    ax.set_title(f"{asset} — Price Forecast Matrix", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
