import pandas as pd
from ST import run_strategy

# 讀取 CSV，指定分隔符、解析 timestamp 欄位，並設為 index
df = pd.read_csv("Bitcoin_2022_1_1-2025_3_24_historical_data_coinmarketcap.csv", delimiter=";", parse_dates=['timestamp'], index_col='timestamp')
df.index = df.index.tz_convert(None)

# 根據時間排序
df.sort_index(inplace=True)

# 檢查前幾筆數據是否正確讀入
print(df.head())
# 呼叫策略回測函數
trade_log = run_strategy(df, enable_dynamic_position_sizing=True, enable_short=True)

# 輸出交易紀錄
for trade in trade_log:
    print(trade)
