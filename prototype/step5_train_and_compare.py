import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = pd.read_csv("final_dataset.csv", parse_dates=["time"])

split = int(len(data) * 0.8)
train, test = data.iloc[:split], data.iloc[split:]

def train_eval(features):
    X_train, y_train = train[features], train["target_price_next"]
    X_test, y_test = test[features], test["target_price_next"]
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    actual_dir = np.sign(y_test.values - test["price"].values)
    pred_dir = np.sign(preds - test["price"].values)
    direction_acc = (actual_dir == pred_dir).mean()

    return preds, mae, rmse, direction_acc

preds_price_only, mae1, rmse1, dir1 = train_eval(["price"])
preds_with_sentiment, mae2, rmse2, dir2 = train_eval(["price", "sentiment_mean"])

print("\n=== RESULTS ===")
print(f"{'Model':<25}{'MAE':>12}{'RMSE':>12}{'Direction Acc':>16}")
print(f"{'Price only':<25}{mae1:>12.2f}{rmse1:>12.2f}{dir1:>16.2%}")
print(f"{'Price + Sentiment':<25}{mae2:>12.2f}{rmse2:>12.2f}{dir2:>16.2%}")
print(f"\nDirectional accuracy change from adding sentiment: {dir2 - dir1:+.2%}")

# Step 7/8: plot actual vs both predictions
plt.figure(figsize=(12, 6))
plt.plot(test["time"], test["target_price_next"], label="Actual price", linewidth=2)
plt.plot(test["time"], preds_price_only, label="Predicted (price only)", linestyle="--")
plt.plot(test["time"], preds_with_sentiment, label="Predicted (price + sentiment)", linestyle="--")
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Actual vs Predicted BTC-USD Price (next hour)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("prediction_chart.png", dpi=120)
print("\nSaved chart to prediction_chart.png")
