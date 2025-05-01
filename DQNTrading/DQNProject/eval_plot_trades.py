import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from env.trading_env_simple import TradingEnv
from agent.dqn_agent import DQNAgent
import mplfinance as mpf
import warnings

# ========== 設定 ==========
MODEL_PATH = r'D:\my_code\DQNTrading\checkpoints\dqn_final.pth'
DATA_PATH = r'D:\my_code\DQNTrading\Data\btc_1h_2024_2025.csv'
ASSET_NAME = 'BTC'

# ========== 載入資料 ==========
test_data = pd.read_csv(DATA_PATH)

# ========== 動態偵測資料頻率 ==========
periods_per_year = periods_per_year = 365 # 預設為年化常用的交易日數

freq_str = "Unknown"
if 'timestamp' in test_data.columns:
    try:
        test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
        time_diffs = test_data['timestamp'].diff().dropna()
        if not time_diffs.empty:
            median_diff = time_diffs.median()
            if median_diff <= pd.Timedelta(minutes=1): periods_per_year = 365*24*60; freq_str="Minute"
            elif median_diff <= pd.Timedelta(hours=1): periods_per_year = 365*24; freq_str="Hourly"
            elif median_diff <= pd.Timedelta(days=1): periods_per_year = 365; freq_str="Daily"
            else: periods_per_year = 1; freq_str=f"Weekly+ ({median_diff})"; warnings.warn("Weekly+ freq")
        else: warnings.warn("No time diffs")
    except Exception as e: warnings.warn(f"Timestamp error: {e}")
else: warnings.warn("No timestamp column")
print(f"Detected data frequency: {freq_str} (Using {periods_per_year:.0f} periods/year)")

# ========== 初始化環境 ==========
env = TradingEnv(test_data, asset_name=ASSET_NAME)

# ========== 初始化Agent ==========
temp_state = env.reset()
state_dim = temp_state.shape[0]
action_dim = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
agent.load_model(MODEL_PATH)
agent.policy_net.eval()
agent.epsilon = 0.0

# ========== 測試並記錄交易 ==========
state = env.reset()
done = False
portfolio_values = [env.total_asset]
entry_signals = []
exit_signals = []
# trade_log_eval = [] # 不再需要，直接使用 env.trade_log

while not done:
    # --- Debug: Print Q-values ---
    with torch.no_grad():
        q_values = agent.policy_net(torch.FloatTensor(state).unsqueeze(0).to(device))
        print(f"Step: {env.current_step}, Q-Values: [Hold: {q_values[0, 0]:.4f}, Long: {q_values[0, 1]:.4f}, Short: {q_values[0, 2]:.4f}]")
    # -----------------------------
    action = agent.select_action(state, evaluation=True)
    print(f"Selected Action: {action}") # Print selected action
    next_state, reward, done, info = env.step(action)
    portfolio_values.append(env.total_asset)

    # 從 env.trade_log 獲取最新交易記錄來更新繪圖信號
    if env.trade_log:
        last_trade = env.trade_log[-1]
        step_idx, trade_type, price, amount, pnl_or_cash = last_trade
        # 檢查這筆交易是否是當前 step 產生的 (避免重複添加舊信號)
        if step_idx == env.current_step - 1: # 因為 step 結束時 current_step 已加 1
            if 'Open' in trade_type:
                entry_signals.append((step_idx, price))
            elif 'Close' in trade_type or 'Stop Loss' in trade_type:
                exit_signals.append((step_idx, price))

    state = next_state

# ========== 畫K棒與交易點 ==========
plot_data = test_data.copy()
if 'timestamp' in plot_data.columns:
    try: plot_data.index = test_data['timestamp']
    except Exception as e: print(f"Index Error: {e}"); plot_data.index = pd.RangeIndex(len(plot_data))
else: plot_data.index = pd.RangeIndex(len(plot_data))

ohlc_data = plot_data[['open', 'high', 'low', 'close', 'volume']]
if not isinstance(ohlc_data.index, pd.DatetimeIndex): print("Warning: Plot index not DatetimeIndex.")

apds = []
buy_signal_prices = pd.Series([np.nan]*len(ohlc_data), index=ohlc_data.index)
for idx, price in entry_signals:
    if idx < len(ohlc_data): buy_signal_prices.iloc[idx] = price # 使用 iloc

sell_signal_prices = pd.Series([np.nan]*len(ohlc_data), index=ohlc_data.index)
for idx, price in exit_signals:
    if idx < len(ohlc_data): sell_signal_prices.iloc[idx] = price # 使用 iloc

if not buy_signal_prices.isnull().all(): apds.append(mpf.make_addplot(buy_signal_prices, type='scatter', markersize=100, marker='^', color='green'))
if not sell_signal_prices.isnull().all(): apds.append(mpf.make_addplot(sell_signal_prices, type='scatter', markersize=100, marker='v', color='red'))

mpf.plot(ohlc_data, type='candle', style='charles', addplot=apds, volume=True,
         title=f'{ASSET_NAME} Trading Behavior Visualization (DQN)', figratio=(16,9), figscale=1.2,
         ylabel='Price', ylabel_lower='Volume')
plt.show()

# ========== 打印詳細交易日誌 ==========
print("\n" + "="*80)
print("Detailed Trading Log:")
print("="*80)
print(f"{'Step':<6} | {'Timestamp':<20} | {'Type':<18} | {'Price':<12} | {'Amount':<15} | {'PnL / Cash After':<18}")
print("-"*80)

for trade in env.trade_log:
    step_idx, trade_type, price, amount, pnl_or_cash = trade
    timestamp_str = ""
    if 'timestamp' in test_data.columns and step_idx < len(test_data):
        timestamp_str = str(test_data.iloc[step_idx]['timestamp']) # 獲取對應時間戳

    if 'Open' in trade_type:
        print(f"{step_idx:<6} | {timestamp_str:<20} | {trade_type:<18} | {price:<12.4f} | {amount:<15.6f} | Cash: {pnl_or_cash:<12.2f}")
    else: # Close or Stop Loss
        print(f"{step_idx:<6} | {timestamp_str:<20} | {trade_type:<18} | {price:<12.4f} | {amount:<15.6f} | PnL: {pnl_or_cash:<+13.2f}") # 使用 '+' 顯示正負號

print("="*80)


# ========== 計算並打印績效指標 ==========
initial_asset = portfolio_values[0]
final_asset = portfolio_values[-1]
total_return = (final_asset - initial_asset) / initial_asset * 100 if initial_asset != 0 else 0

peak = portfolio_values[0]
max_drawdown = 0
for value in portfolio_values:
    if value > peak: peak = value
    safe_peak = peak if peak != 0 else 1e-8
    drawdown = (safe_peak - value) / safe_peak
    if drawdown > max_drawdown: max_drawdown = drawdown

portfolio_array = np.array(portfolio_values)
safe_denominators = np.where(portfolio_array[:-1] == 0, 1e-8, portfolio_array[:-1])
period_returns = (portfolio_array[1:] - portfolio_array[:-1]) / safe_denominators

if len(period_returns) > 0:
    risk_free_rate = 3.0
    annualized_return = np.mean(period_returns) * periods_per_year
    annualized_volatility = np.std(period_returns) * np.sqrt(periods_per_year)
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = excess_return / (annualized_volatility + 1e-8)
else:
    annualized_return = 0.0
    annualized_volatility = 0.0
    sharpe_ratio = 0.0

print("\n" + "="*60)
print("Performance Metrics:")
print("="*60)
print(f"資產初始值 Initial Asset:    ${initial_asset:.2f}")
print(f"資產最終值 Final Asset:      ${final_asset:.2f}")
print(f"總報酬率 Total Return:       {total_return:.2f}%")
print(f"最大回撤 Max Drawdown:       {max_drawdown:.2%}")
print(f"年化報酬率 Annualized Return: {annualized_return:.2%}")
print(f"年化波動率 Annualized Vol:    {annualized_volatility:.2%}")
print(f"夏普比率 Sharpe Ratio:        {sharpe_ratio:.2f}")
print(f"總交易次數 Total Trades:     {len(env.trade_log)}") # 使用 env.trade_log
print("="*60)
