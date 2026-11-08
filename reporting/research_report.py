from datetime import datetime, timezone

import pandas as pd


def generate_evaluation_report(summary: pd.DataFrame, asset: str) -> str:
    """Builds the required evaluation report (Markdown) with MAE/RMSE/direction accuracy
    plus Sharpe Ratio, Sortino Ratio and Max Drawdown per model/feature-set/horizon,
    and the key research question: does sentiment (and news) improve on price-only."""
    now = datetime.now(timezone.utc).isoformat()

    lines = [
        f"# {asset} — Model Evaluation Report",
        "",
        f"Generated: {now}",
        "",
        "## Methodology",
        "- Walk-forward validation with an expanding training window; test folds always come strictly after training data.",
        "- Three feature sets compared: price-only, price+X-sentiment, price+X-sentiment+news.",
        "- Models: XGBoost, LightGBM, evaluated at horizons n+1..n+5.",
        "- Trading metrics (Sharpe, Sortino, Max Drawdown) come from a simple long/short strategy: "
        "position direction = sign of predicted price change, realized return = actual price change over that horizon.",
        "",
        "## Results",
        "",
    ]

    for model_name in sorted(summary["model"].unique()):
        lines.append(f"### Model: {model_name}")
        lines.append("")
        model_summary = summary[summary["model"] == model_name]
        cols = [
            "feature_set",
            "horizon",
            "mae",
            "rmse",
            "direction_accuracy",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "delta_accuracy_vs_price_only",
        ]
        table = model_summary[cols].sort_values(["horizon", "feature_set"])
        lines.append(table.to_markdown(index=False, floatfmt=".4f"))
        lines.append("")

    lines.append("## Key research question")
    lines.append("")
    lines.append(
        "`delta_accuracy_vs_price_only` is the directional-accuracy improvement from adding "
        "X sentiment (and news) over price alone. If this is not consistently positive across "
        "horizons and folds, there is no evidence yet that sentiment adds forecasting value for this asset/period."
    )

    return "\n".join(lines)
