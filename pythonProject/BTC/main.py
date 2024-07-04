import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import ta
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# Load environment variables and API key
load_dotenv()
api_key = os.getenv("BINANCE_API_KEY")

# Function to get historical data from Binance API
def get_binance_klines(symbol="BTCUSDT", interval="1d", limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    headers = {"X-MBX-APIKEY": api_key}
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    df = pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    return df[["open", "high", "low", "close", "volume"]]

# Get data
df = get_binance_klines(limit=1000)  # Increase limit to get more historical data

# Data preprocessing and feature engineering
df["price_change"] = df["close"].pct_change()
df["7_day_ma"] = df["close"].rolling(window=7).mean()
df["30_day_ma"] = df["close"].rolling(window=30).mean()
df["RSI"] = ta.momentum.rsi(df["close"], window=14)
df["MACD"] = ta.trend.macd_diff(df["close"])
df["Bollinger_High"] = ta.volatility.bollinger_hband(df["close"])
df["Bollinger_Low"] = ta.volatility.bollinger_lband(df["close"])
df.bfill(inplace=True)  # Updated method to avoid FutureWarning

features = [
    "open", "high", "low", "close", "volume",
    "price_change", "7_day_ma", "30_day_ma",
    "RSI", "MACD", "Bollinger_High", "Bollinger_Low"
]
data = df[features]

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Function to create sequences
def create_sequences(data, time_steps=120):  # Increase time_steps for longer sequences
    sequences = []
    labels = []
    for i in range(len(data) - time_steps):
        sequences.append(data[i : i + time_steps])
        labels.append(data[i + time_steps, 3])  # Correct index for 'close' price
    return np.array(sequences), np.array(labels)

time_steps = 120  # Increase time steps
X, y = create_sequences(scaled_data, time_steps)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Ensure that the GPU is used
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected.")

# Efficient data input pipeline
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.cache().shuffle(buffer_size=1024).batch(64).prefetch(tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
val_dataset = val_dataset.batch(64).prefetch(tf.data.experimental.AUTOTUNE)

# Build the LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(time_steps, X_train.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=200)  # Increase epochs for longer training

# Evaluate the model
train_loss = model.evaluate(train_dataset, verbose=0)
test_loss = model.evaluate(val_dataset, verbose=0)
print(f"Train Loss: {train_loss}")
print(f"Test Loss: {test_loss}")

# Predict on the validation dataset
predictions = model.predict(val_dataset)
predictions = scaler.inverse_transform(
    np.concatenate(
        (np.zeros((predictions.shape[0], data.shape[1] - 1)), predictions), axis=1
    )
)[:, -1]

# Plot the predictions vs true prices
plt.figure(figsize=(10, 6))
plt.plot(df.index[train_size + time_steps:], scaler.inverse_transform(scaled_data)[train_size + time_steps:, 3], label="True Price")
plt.plot(df.index[train_size + time_steps:], predictions, label="Predicted Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.title("BTCUSDT Price Prediction")
plt.show()

# Function to fetch the latest kline data
def get_latest_binance_klines(symbol="BTCUSDT", interval="1d", limit=120):
    url = "https://api.binance.com/api/v3/klines"
    headers = {"X-MBX-APIKEY": api_key}
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    df = pd.DataFrame(
        data,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    return df[["open", "high", "low", "close", "volume"]]

# Get the latest data (last 120 days for example)
latest_df = get_latest_binance_klines(limit=120)

# Feature engineering
latest_df["price_change"] = latest_df["close"].pct_change()
latest_df["7_day_ma"] = latest_df["close"].rolling(window=7).mean()
latest_df["30_day_ma"] = latest_df["close"].rolling(window=30).mean()
latest_df["RSI"] = ta.momentum.rsi(latest_df["close"], window=14)
latest_df["MACD"] = ta.trend.macd_diff(latest_df["close"])
latest_df["Bollinger_High"] = ta.volatility.bollinger_hband(latest_df["close"])
latest_df["Bollinger_Low"] = ta.volatility.bollinger_lband(latest_df["close"])
latest_df.bfill(inplace=True)  # Fill any NaNs

# Select the same features
latest_data = latest_df[features]

# Scale the latest data
scaled_latest_data = scaler.transform(latest_data)

# Use the last `time_steps` data points to create the sequence
latest_sequence = np.array([scaled_latest_data[-time_steps:]])

# Predict the next price
next_price_scaled = model.predict(latest_sequence)
next_price_scaled = next_price_scaled.flatten()  # Flatten to get a single value

# Inverse transform to get the actual price
# We need to pad with zeros to use the scaler's inverse_transform
next_price_unscaled = scaler.inverse_transform(
    np.concatenate((np.zeros((1, data.shape[1] - 1)), next_price_scaled.reshape(1, -1)), axis=1)
)[0, -1]

print(f"Predicted next close price: {next_price_unscaled:.2f}")

# Function to predict future steps
def predict_future_steps(model, latest_data, steps, scaler, time_steps=120):
    current_sequence = latest_data[-time_steps:].copy()
    future_predictions = []

    for _ in range(steps):
        next_price_scaled = model.predict(np.array([current_sequence]))
        next_price_scaled = next_price_scaled.flatten()

        next_price_unscaled = scaler.inverse_transform(
            np.concatenate((np.zeros((1, data.shape[1] - 1)), next_price_scaled.reshape(1, -1)), axis=1)
        )[0, -1]

        future_predictions.append(next_price_unscaled)

        # Add the new prediction and remove the oldest data point to update the sequence
        current_sequence = np.vstack((current_sequence[1:], next_price_scaled))

    return future_predictions

# Predict the next 10 days (or any number of steps)
future_steps = 10
future_predictions = predict_future_steps(model, scaled_latest_data, future_steps, scaler)

print("Future Predictions:")
for i, price in enumerate(future_predictions, 1):
    print(f"Day {i}: {price:.2f}")

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df.index[-200:], scaler.inverse_transform(scaled_data)[-200:, 3], label="Historical Close Price")
future_dates = pd.date_range(df.index[-1], periods=future_steps + 1, freq="D")[1:]
plt.plot(future_dates, future_predictions, label="Future Predictions", linestyle='dashed')
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.title("BTCUSDT Price Prediction")
plt.show()
