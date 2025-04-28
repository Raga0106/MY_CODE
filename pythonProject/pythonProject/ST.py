import pandas as pd
import numpy as np


def run_strategy(df,
                 use_date_filter=True,
                 backtest_start=pd.Timestamp("2022-01-01"),
                 backtest_end=pd.Timestamp("2222-01-01"),
                 enable_dynamic_position_sizing=False,
                 enable_short=False):
    """
    模擬原 PineScript 策略：
      - 動態 ATR 停損
      - 多組 RSI 計算與條件判斷
      - Williams Vix Fix 指標與布林通道相關判斷
      - 長單 / 空單進出場邏輯與動態持倉量調整

    參數:
      df: 包含歷史資料的 DataFrame，必須有欄位 'open','high','low','close'
      use_date_filter: 是否啟用回測時間範圍
      backtest_start, backtest_end: 回測時間區間（Timestamp 格式）
      enable_dynamic_position_sizing: 是否啟用動態持倉調整
      enable_short: 是否啟用做空策略
    回傳:
      trade_log: 記錄每次交易進出場的 log (tuple: timestamp, 訊息, 價格)
    """
    # --------------------------------------------------------------------
    # 參數設定（參考原始 input）
    # --------------------------------------------------------------------
    timeWindow = 5  # 固定觀察窗，5 根K線
    shortMaLength = 200  # 做空時的均線參數

    # SuperATR 止損參數
    atr_period_short = 3
    atr_period_long = 7
    momentum_period = 7
    atr_stop_multiplier = 2.0

    # 止盈參數
    min_profit_percent = 0.5

    # 動態持倉量調整參數
    atrThreshold = 1.0
    slopeThreshold = 0.1
    basePositionSize = 1
    maxExtraPosition = 5
    bb_distance_threshold = 0.05

    # 布林帶參數
    bb_length = 20
    bb_mult = 2.0
    bb_ma_length = 5
    distance_threshold = 0.05

    # RSI 參數
    periods_fast = [3, 5, 8, 10, 12, 15]  # RSI1 ~ RSI6
    periods_slow = [30, 35, 40, 45, 50, 60]  # RSI7 ~ RSI12

    # CM_Williams_Vix_Fix 參數
    pd_param = 22
    bbl = 20
    mult_param = 2.0
    lb = 50
    ph = 0.85
    pl = 1.01

    # --------------------------------------------------------------------
    # 定義輔助函數
    # --------------------------------------------------------------------
    def calculate_true_range(df):
        tr1 = df['high'] - df['low']
        tr2 = np.abs(df['high'] - df['close'].shift(1))
        tr3 = np.abs(df['low'] - df['close'].shift(1))
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    def rsi_period(series, length):
        """利用 Wilder 平滑法計算 RSI"""
        delta = series.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        # 使用 ewm(alpha=1/length) 模擬 Wilder 平滑
        up_ewm = up.ewm(alpha=1 / length, adjust=False).mean()
        down_ewm = down.ewm(alpha=1 / length, adjust=False).mean()
        rs = up_ewm / down_ewm
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calc_linreg(series, window):
        """計算線性回歸斜率，採用 rolling window"""
        return series.rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)

    # --------------------------------------------------------------------
    # 計算各項指標
    # --------------------------------------------------------------------
    df['tr'] = calculate_true_range(df)
    df['short_atr'] = df['tr'].rolling(window=atr_period_short).mean()
    df['long_atr'] = df['tr'].rolling(window=atr_period_long).mean()
    df['momentum'] = df['close'] - df['close'].shift(momentum_period)
    df['stdev_close'] = df['close'].rolling(window=momentum_period).std()
    df['normalized_momentum'] = np.where(df['stdev_close'] != 0, df['momentum'] / df['stdev_close'], 0)
    df['momentum_factor'] = df['normalized_momentum'].abs()
    df['adaptive_atr'] = (df['short_atr'] * df['momentum_factor'] + df['long_atr']) / (1 + df['momentum_factor'])

    # 布林帶計算
    df['basis'] = df['close'].rolling(window=bb_length).mean()
    df['dev'] = bb_mult * df['close'].rolling(window=bb_length).std()
    df['upper_bb'] = df['basis'] + df['dev']
    df['lower_bb'] = df['basis'] - df['dev']
    df['bb_distance'] = df['upper_bb'] - df['lower_bb']

    # 多組 RSI 計算 (快組與慢組)
    for i, p in enumerate(periods_fast):
        df[f'rsi_fast_{i}'] = rsi_period(df['close'], p)
    for i, p in enumerate(periods_slow):
        df[f'rsi_slow_{i}'] = rsi_period(df['close'], p)

    # 快組統計
    rsi_fast_cols = [f'rsi_fast_{i}' for i in range(len(periods_fast))]
    df['sum_fast'] = df[rsi_fast_cols].sum(axis=1)
    df['sum_sq_fast'] = (df[rsi_fast_cols] ** 2).sum(axis=1)
    df['mean_fast'] = np.sqrt(df['sum_sq_fast'] / len(rsi_fast_cols))
    df['variance_fast'] = (df['sum_sq_fast'] - (df['sum_fast'] ** 2) / len(rsi_fast_cols)) / len(rsi_fast_cols)
    df['sigma_fast'] = np.sqrt(df['variance_fast'])

    # 慢組統計
    rsi_slow_cols = [f'rsi_slow_{i}' for i in range(len(periods_slow))]
    df['sum_slow'] = df[rsi_slow_cols].sum(axis=1)
    df['sum_sq_slow'] = (df[rsi_slow_cols] ** 2).sum(axis=1)
    df['mean_slow'] = np.sqrt(df['sum_sq_slow'] / len(rsi_slow_cols))
    df['variance_slow'] = (df['sum_sq_slow'] - (df['sum_slow'] ** 2) / len(rsi_slow_cols)) / len(rsi_slow_cols)
    df['sigma_slow'] = np.sqrt(df['variance_slow'])

    # 標準化 sigma (以 lookback 期數計算)
    lookback = 100
    df['lowest_fast'] = df['sigma_fast'].rolling(window=lookback).min()
    df['highest_fast'] = df['sigma_fast'].rolling(window=lookback).max()
    df['lowest_slow'] = df['sigma_slow'].rolling(window=lookback).min()
    df['highest_slow'] = df['sigma_slow'].rolling(window=lookback).max()
    df['normalized_sigma_fast'] = np.where(df['highest_fast'] != df['lowest_fast'],
                                           100 * (df['sigma_fast'] - df['lowest_fast']) / (
                                                       df['highest_fast'] - df['lowest_fast']),
                                           0)
    df['normalized_sigma_slow'] = np.where(df['highest_slow'] != df['lowest_slow'],
                                           100 * (df['sigma_slow'] - df['lowest_slow']) / (
                                                       df['highest_slow'] - df['lowest_slow']),
                                           0)
    df['tight_fast'] = 100 - df['normalized_sigma_fast']
    df['tight_slow'] = 100 - df['normalized_sigma_slow']
    df['rsidifference'] = df['mean_fast'] - df['mean_slow']
    df['rsidifference_tight'] = df['rsidifference'] * df['tight_fast'] * df['tight_slow'] / 10000

    # RSI 條件：定義綠柱與紅柱（參考前一根、前二根的比較）
    df['RSIisGreenBar'] = (df['rsidifference_tight'] > 0) & (df['rsidifference_tight'].shift(1) <= 0) & (
            ((df['tight_fast'] > df['tight_fast'].shift(1)) | (df['tight_slow'] > df['tight_slow'].shift(1))) |
            ((df['tight_fast'].shift(1) < df['tight_fast'].shift(2)) | (
                        df['tight_slow'].shift(1) < df['tight_slow'].shift(2)))
    )
    df['RSIisRedBar'] = (df['rsidifference_tight'] < 0) & (df['rsidifference_tight'].shift(1) >= 0) & (
            ((df['tight_fast'] > df['tight_fast'].shift(1)) | (df['tight_slow'] > df['tight_slow'].shift(1))) |
            ((df['tight_fast'].shift(1) < df['tight_fast'].shift(2)) | (
                        df['tight_slow'].shift(1) < df['tight_slow'].shift(2)))
    )

    # Williams Vix Fix 指標計算
    df['highest_close_pd'] = df['close'].rolling(window=pd_param).max()
    df['wvf'] = ((df['highest_close_pd'] - df['low']) / df['highest_close_pd']) * 100
    df['sDev'] = mult_param * df['wvf'].rolling(window=bbl).std()
    df['midLine'] = df['wvf'].rolling(window=bbl).mean()
    df['lowerBand'] = df['midLine'] - df['sDev']
    df['upperBand'] = df['midLine'] + df['sDev']
    df['rangeHigh'] = df['wvf'].rolling(window=lb).max() * ph
    df['rangeLow'] = df['wvf'].rolling(window=lb).min() * pl

    # 定義 VixFix 顏色/條件（此處僅以條件判斷，不做視覺化）
    df['isGreenBar'] = (df['wvf'] >= df['upperBand']) | (df['wvf'] >= df['rangeHigh'])
    # crossunder 判斷：前一根在 lowerBand 之上、目前下穿 lowerBand
    df['isRedBar'] = (df['wvf'].shift(1) > df['lowerBand'].shift(1)) & (df['wvf'] < df['lowerBand'])

    # 計算短單均線
    df['maShort'] = df['close'].rolling(window=shortMaLength).mean()
    # 趨勢斜率 (以最近20根 K 線的線性回歸斜率)
    df['trendSlope'] = calc_linreg(df['close'], 20)

    # --------------------------------------------------------------------
    # 模擬交易狀態變數初始化
    # --------------------------------------------------------------------
    position = 0  # 持倉：正數代表多單，負數代表空單
    position_avg_price = 0.0  # 持倉平均價
    trade_log = []  # 記錄交易訊息

    # 多單相關狀態
    entryBarHigh = np.nan
    entryLow = np.nan
    waitForEntry = False
    waitForExit = False
    stopLossPrice = np.nan
    tradeId = 0
    greenBarIndex = 0
    redBarIndex = 0

    # 空單相關狀態
    shortBarHigh = np.nan
    shortBarLow = np.nan
    waitForShortEntry = False
    waitForShortExit = False
    shortStopLossPrice = np.nan
    shortTradeId = 0

    maxPositionSize = basePositionSize

    # --------------------------------------------------------------------
    # 逐筆回測模擬 (依據 DataFrame 的 index 遍歷)
    # --------------------------------------------------------------------
    for i in range(len(df)):
        # 取當前 bar 的資料
        row = df.iloc[i]
        timestamp = row.name  # 假設 index 為 timestamp
        inTradeWindow = (not use_date_filter) or ((timestamp >= backtest_start) and (timestamp < backtest_end))

        # 動態持倉量調整
        if enable_dynamic_position_sizing:
            isHighVolatility = row['bb_distance'] > bb_distance_threshold
            isStrongTrend = abs(row['trendSlope']) > slopeThreshold
            if isHighVolatility and isStrongTrend:
                maxPositionSize = basePositionSize + maxExtraPosition
            elif isHighVolatility or isStrongTrend:
                maxPositionSize = basePositionSize + maxExtraPosition / 2
            else:
                maxPositionSize = basePositionSize

        # ----------------------------------------------------------------
        # 15) 更新「看多」關鍵 K 線
        # ----------------------------------------------------------------
        if (row['isGreenBar'] or row['RSIisGreenBar']) and not (row['isRedBar'] or row['RSIisRedBar']):
            entryBarHigh = row['high']
            entryLow = row['low']
            waitForEntry = True
            waitForExit = False
            greenBarIndex = i

        # 若遇紅柱且已有多單，準備平倉
        if (row['isRedBar'] or row['RSIisRedBar']) and (position > 0):
            exitBarLow = row['low']
            exitBarHigh = row['high']
            waitForExit = True
            waitForEntry = False
            entryBarHigh = np.nan
            entryLow = np.nan
            redBarIndex = i

        # ----------------------------------------------------------------
        # 16) 更新「看空」關鍵 K 線（無持空單或空單數量<=0時才偵測）
        # ----------------------------------------------------------------
        if (row['isRedBar'] or row['RSIisRedBar']) and (position <= 0):
            shortBarHigh = row['high']
            shortBarLow = row['low']
            waitForShortEntry = True
            waitForShortExit = False
            waitForEntry = False

        if (row['isGreenBar'] or row['RSIisGreenBar']) and (position < 0) and not (
                row['isRedBar'] or row['RSIisRedBar']):
            waitForShortExit = True
            waitForShortEntry = False

        # ----------------------------------------------------------------
        # 17) 定義「突破」函數 (多單/空單)
        # ----------------------------------------------------------------
        def isBreakHigh(refHigh, current_index, green_index):
            return (row['close'] > refHigh) and (current_index <= green_index + timeWindow)

        def isBreakLow(refLow, current_index, red_index):
            return (row['close'] < refLow) and (current_index <= red_index + timeWindow)

        def isBreakShort(refLow, current_index, red_index):
            return (row['close'] < refLow) and (current_index <= red_index + timeWindow)

        # ----------------------------------------------------------------
        # 18) 計算當前獲利百分比
        # ----------------------------------------------------------------
        current_profit_percent = 0.0
        current_profit_percent_short = 0.0
        if position > 0:
            current_profit_percent = ((row['close'] - position_avg_price) / position_avg_price) * 100
        elif position < 0:
            current_profit_percent_short = ((position_avg_price - row['close']) / position_avg_price) * 100

        # ----------------------------------------------------------------
        # 19) 交易條件 (多單)
        # ----------------------------------------------------------------
        # 注意：這裡以前一筆 mean_slow 值做比較（i>0時）
        longCondition = (waitForEntry and
                         (not np.isnan(entryBarHigh)) and
                         isBreakHigh(entryBarHigh, i, greenBarIndex) and
                         (i > 0 and row['mean_slow'] > df['mean_slow'].iloc[i - 1]))

        takeProfitCondition = (waitForExit and (position > 0) and
                               (not np.isnan(exitBarLow)) and
                               isBreakLow(exitBarLow, i, redBarIndex))

        stopLossCondition = (position > 0) and (row['low'] < stopLossPrice)

        # ----------------------------------------------------------------
        # 20) 交易條件 (空單)
        # ----------------------------------------------------------------
        shortCondition = (waitForShortEntry and
                          (not np.isnan(shortBarLow)) and
                          isBreakShort(shortBarLow, i, redBarIndex) and
                          (i > 0 and row['mean_slow'] < df['mean_slow'].iloc[i - 1]))

        # ----------------------------------------------------------------
        # 22) 交易執行邏輯 - 多單
        # ----------------------------------------------------------------
        if longCondition and inTradeWindow and (not (row['isRedBar'] or row['RSIisRedBar'])):
            if (not enable_dynamic_position_sizing) or (abs(position) < maxPositionSize):
                tradeId += 1
                # 進場：多單
                position = 1
                position_avg_price = row['close']
                stopLossPrice = row['close'] - (row['adaptive_atr'] * atr_stop_multiplier)
                waitForEntry = False
                trade_log.append((timestamp, f"Long Entry (ID {tradeId})", row['close']))

        # 平倉條件：獲利或突破 exitBarLow
        if takeProfitCondition and inTradeWindow and position > 0:
            trade_log.append((timestamp, f"Long Exit Take Profit ({current_profit_percent:.2f}%)", row['close']))
            position = 0
            waitForExit = False
            tradeId = 0

        if stopLossCondition and inTradeWindow and position > 0:
            trade_log.append((timestamp, "Long Exit Stop Loss", row['close']))
            position = 0
            waitForExit = False
            tradeId = 0

        # ----------------------------------------------------------------
        # 23) 交易執行邏輯 - 空單 (僅示範)
        # ----------------------------------------------------------------
        if enable_short and row['close'] < row['maShort']:
            if shortCondition and inTradeWindow:
                if (not enable_dynamic_position_sizing) or (abs(position) < maxPositionSize):
                    shortTradeId += 1
                    position = -1
                    position_avg_price = row['close']
                    shortStopLossPrice = row['close'] + (row['adaptive_atr'] * atr_stop_multiplier)
                    waitForShortEntry = False
                    trade_log.append((timestamp, f"Short Entry (ID {shortTradeId})", row['close']))

            if waitForShortExit and position < 0 and inTradeWindow and (not np.isnan(entryBarHigh)) and isBreakHigh(
                    entryBarHigh, i, greenBarIndex):
                trade_log.append(
                    (timestamp, f"Short Exit Take Profit ({current_profit_percent_short:.2f}%)", row['close']))
                position = 0
                waitForShortExit = False
                shortTradeId = 0

            if (position < 0) and (row['high'] > shortStopLossPrice) and inTradeWindow:
                trade_log.append((timestamp, "Short Exit Stop Loss", row['close']))
                position = 0
                waitForShortExit = False
                shortTradeId = 0

    return trade_log

# -----------------------------------------------------------
# 使用範例
# -----------------------------------------------------------
# 假設您已有歷史資料檔案，且資料包含 timestamp、open、high、low、close 等欄位
# df = pd.read_csv("historical_data.csv", parse_dates=['timestamp'], index_col='timestamp')
# trades = run_strategy(df, enable_dynamic_position_sizing=True, enable_short=True)
# for t in trades:
#     print(t)
