import requests
import pandas as pd
import time

def get_historical_klines(symbol, interval, start_str, end_str, filename):
    """從 Binance API 取得指定時間範圍的 K 線數據，並儲存為 CSV"""
    url = "https://api.binance.com/api/v3/klines"
    start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)  # 轉換成毫秒時間戳
    end_ts = int(pd.Timestamp(end_str).timestamp() * 1000)
    limit = 1000
    all_data = []
    
    while start_ts < end_ts:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ts,
            "endTime": end_ts,
            "limit": limit
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        if not data:
            break

        all_data.extend(data)
        
        # 更新下一個查詢的 startTime（最後一條 K 線的時間戳記 +1 毫秒）
        start_ts = data[-1][0] + 1
        time.sleep(0.5)  # 避免 API 頻率限制
    
    # 轉換資料為 Pandas DataFrame
    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])

    # 轉換時間戳
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # 選擇需要的列
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    # 保存到 CSV
    df.to_csv(filename, index=False)
    print(f"數據已保存到 {filename}")


# EX # EX # EX # EX # EX # EX # EX # EX # EX # EX # EX # EX # EX # EX # EX # EX # EX # EX # EX # EX # EX # EX #  

# 取得 BTC/USDT 2020-2024 日 K 線數據，並儲存為 CSV
get_historical_klines("BTCUSDT", "1h", "2020-01-01", "2024-01-01", "btc_1h_2020_2024.csv")
get_historical_klines("BTCUSDT", "1h", "2024-01-01", "2025-01-01", "btc_1h_2024_2025.csv")


# 取得 ETH/USDT 2020-2024 日 K 線數據，並儲存為 CSV
get_historical_klines("ETHUSDT", "1h", "2020-01-01", "2024-01-01", "eth_1h_2020_2024.csv")
get_historical_klines("ETHUSDT", "1h", "2024-01-01", "2025-01-01", "eth_1h_2024_2025.csv")

# 取得 SOL/USDT 2020-2024 4小時 K 線數據，並儲存為 CSV
get_historical_klines("SOLUSDT", "1h", "2020-01-01", "2024-01-01", "sol_1h_2020_2024.csv")
get_historical_klines("SOLUSDT", "1h", "2024-01-01", "2025-01-01", "sol_1h_2024_2025.csv")

