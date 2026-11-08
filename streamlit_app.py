"""Interactive dashboard presenting the project deliverable end to end.

Wraps the exact same pipeline functions demo_end_to_end.py calls (no
duplicated logic) behind a UI: a "Live Demo" tab that runs the live
scrape -> ingest -> train -> predict pipeline on a button click and renders
out-of-sample metrics, actual-vs-predicted and sentiment-vs-price as
interactive Plotly charts, plus "Historical Backtest" tabs that display the
already-generated evaluation reports and charts from run_historical_backtest.py
and run_historical_backtest_combined.py.

Run with: streamlit run streamlit_app.py
"""
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import ASSET_TICKER, HORIZONS
from database.connection import init_db
from features.feature_pipeline import FEATURE_SETS
from models.inference import generate_prediction_matrix
from models.multi_horizon import predict_multi_horizon_price, train_multi_horizon
from models.xgboost_model import build_xgboost_model
from processing.dataset_builder import build_and_store_dataset
from validation.leakage_checks import assert_no_overlap
from validation.metrics import forecast_metrics
from validation.walk_forward import chronological_split

st.set_page_config(page_title="X Sentiment + Price Forecast", layout="wide", page_icon="chart")


def actual_vs_predicted_chart(test: pd.DataFrame, test_preds: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test["timestamp_utc"], y=test["price"], name="Actual price", line=dict(color="black", width=2)))
    for h in HORIZONS:
        if h == 0:
            continue
        fig.add_trace(go.Scatter(x=test["timestamp_utc"], y=test_preds[f"n{h}"], name=f"Predicted n+{h}", line=dict(dash="dash"), opacity=0.7))
    fig.update_layout(title="Actual vs Predicted price (out-of-sample test window)", xaxis_title="Time", yaxis_title="Price (USD)", height=450)
    return fig


def sentiment_vs_price_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp_utc"], y=df["price"], name="Price", line=dict(color="royalblue"), yaxis="y1"))
    fig.add_trace(go.Scatter(x=df["timestamp_utc"], y=df["sentiment_mean"], name="X sentiment", line=dict(color="seagreen"), yaxis="y2"))
    fig.update_layout(
        title="X sentiment vs price over time",
        xaxis_title="Time",
        yaxis=dict(title="Price (USD)", color="royalblue"),
        yaxis2=dict(title="Sentiment score", color="seagreen", overlaying="y", side="right"),
        height=450,
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def prediction_matrix_chart(matrix: pd.DataFrame) -> go.Figure:
    colors = ["seagreen" if d == "up" else "crimson" if d == "down" else "slategray" for d in matrix["direction"]]
    fig = go.Figure(go.Bar(x=matrix["horizon"], y=matrix["change_pct"], marker_color=colors, text=matrix["change_pct"].map(lambda v: f"{v:+.2f}%"), textposition="outside"))
    fig.update_layout(title=f"{ASSET_TICKER} predicted % change by horizon", yaxis_title="Predicted change (%)", height=350)
    return fig


st.title("X Sentiment + Price Prediction")
st.caption("CrowdWisdomTrading assessment -- predicts BTC-USD price at n+0..n+5 from X sentiment and price only, with walk-forward out-of-sample validation.")

tab_live, tab_hist, tab_combined = st.tabs(["Live Demo", "Historical Backtest (2021-2023)", "Combined Backtest (2014-2023)"])

with tab_live:
    st.subheader("Run the full pipeline: scrape -> ingest -> join -> train -> predict")
    st.caption("Uses cached X posts by default (no Apify spend). Delete data/raw/x_posts_live_cache.csv to force a fresh paid run.")

    if st.button("Run live demo pipeline", type="primary"):
        import main as live_pipeline

        status = st.status("Running pipeline...", expanded=True)

        status.write("Step 1-2: scraping X posts + ingesting news + price")
        init_db()
        live_pipeline.ingest_price()
        posts = live_pipeline.ingest_posts(use_cache=True)
        news = live_pipeline.ingest_news()
        status.write(f"-> {len(posts)} real X posts, {len(news)} news articles")

        status.write("Step 3-4: joining in the database + feature engineering")
        dataset = build_and_store_dataset()
        status.write(f"-> {len(dataset)} feature rows built")

        status.write("Step 5: training + walk-forward out-of-sample validation")
        feature_cols = FEATURE_SETS["price_plus_sentiment"]
        train, val, test = chronological_split(dataset)
        assert_no_overlap(train, test)
        models = train_multi_horizon(train, feature_cols, build_xgboost_model)
        test_preds = predict_multi_horizon_price(models, test, feature_cols)

        metrics_rows = []
        for h in HORIZONS:
            if h == 0:
                continue
            m = forecast_metrics(test[f"future_price_n{h}"].values, test_preds[f"n{h}"].values, test["price"].values)
            metrics_rows.append({"Horizon": f"n+{h}", "MAE": m["mae"], "RMSE": m["rmse"], "Direction Accuracy": m["direction_accuracy"]})
        metrics_df = pd.DataFrame(metrics_rows)
        status.write("-> out-of-sample metrics computed on data the model never trained on")

        status.write("Step 6: training final model + generating the n+0..n+5 forecast")
        final_models = train_multi_horizon(dataset, feature_cols, build_xgboost_model)
        latest_row = dataset.tail(1)
        matrix = generate_prediction_matrix(final_models, latest_row, feature_cols)
        status.update(label="Pipeline complete", state="complete", expanded=False)

        st.markdown(f"### {ASSET_TICKER} forecast as of `{latest_row['timestamp_utc'].iloc[0]}`")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(
                matrix, hide_index=True, use_container_width=True,
                column_config={
                    "predicted_price": st.column_config.NumberColumn("predicted_price", format="%.2f"),
                    "change_pct": st.column_config.NumberColumn("change_pct", format="%+.2f%%"),
                },
            )
        with c2:
            st.plotly_chart(prediction_matrix_chart(matrix), use_container_width=True)

        st.markdown("### Out-of-sample performance (honest -- never seen during training)")
        st.dataframe(
            metrics_df, hide_index=True, use_container_width=True,
            column_config={
                "MAE": st.column_config.NumberColumn("MAE", format="%.2f"),
                "RMSE": st.column_config.NumberColumn("RMSE", format="%.2f"),
                "Direction Accuracy": st.column_config.NumberColumn("Direction Accuracy", format="percent"),
            },
        )
        avg_dir_acc = metrics_df["Direction Accuracy"].mean()
        st.caption(
            f"Average out-of-sample direction accuracy: {avg_dir_acc:.1%} "
            f"({'above' if avg_dir_acc > 0.52 else 'near'} the 50% coin-flip baseline -- "
            "see README for the honest interpretation of what this means)."
        )

        st.plotly_chart(actual_vs_predicted_chart(test, test_preds), use_container_width=True)
        st.plotly_chart(sentiment_vs_price_chart(dataset), use_container_width=True)
    else:
        st.info("Click the button above to run the live pipeline end to end and see real output.")

with tab_hist:
    st.subheader("Historical backtest: daily BTC-USD, 2021-2023 (Kaggle tweet sample)")
    report_path = "data/reports/historical_evaluation_report.md"
    img_path = "data/reports/historical_sentiment_vs_price.png"
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("Run `python run_historical_backtest.py` first to generate this report.")
    if os.path.exists(img_path):
        st.image(img_path, caption="Sentiment vs price, 2021-2023")

with tab_combined:
    st.subheader("Combined backtest: daily BTC-USD, 2014-2021 + 2021-2023")
    report_path = "data/reports/historical_evaluation_report_combined.md"
    img_path = "data/reports/historical_sentiment_vs_price_combined.png"
    if os.path.exists(report_path):
        with open(report_path, encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("Run `python run_historical_backtest_combined.py` first to generate this report (may still be running).")
    if os.path.exists(img_path):
        st.image(img_path, caption="Sentiment vs price, 2014-2023 combined")
