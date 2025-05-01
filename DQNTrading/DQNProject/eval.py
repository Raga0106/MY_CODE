import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from env.trading_env_simple import TradingEnv
from agent.dqn_agent import DQNAgent
import warnings # 引入 warnings

# ========== 設定 ==========
MODEL_PATH = r'D:\my_code\DQNTrading\checkpoints\dqn_final.pth'
DATA_PATH = r'D:\my_code\DQNTrading\Data\btc_1h_2024_2025.csv'
ASSET_NAME = 'BTC'

# ========== 載入資料 ==========
test_data = pd.read_csv(DATA_PATH)

# ========== 動態偵測資料頻率 ==========
periods_per_year = 365 # 預設為年化常用的交易日數
freq_str = "Unknown"
if 'timestamp' in test_data.columns:
    try:
        # 嘗試轉換 timestamp，如果失敗則使用預設值
        test_data['timestamp'] = pd.to_datetime(test_data['timestamp'])
        time_diffs = test_data['timestamp'].diff().dropna()
        if not time_diffs.empty:
            median_diff = time_diffs.median()

            # 根據中位數時間差判斷頻率並計算每年期數
            if median_diff <= pd.Timedelta(minutes=1):
                periods_per_year = 365 * 24 * 60
                freq_str = "Minute"
            elif median_diff <= pd.Timedelta(minutes=5):
                 periods_per_year = 365 * 24 * (60 / (median_diff.total_seconds() / 60))
                 freq_str = f"{int(median_diff.total_seconds() / 60)} Minutes"
            elif median_diff <= pd.Timedelta(minutes=15):
                 periods_per_year = 365 * 24 * (60 / (median_diff.total_seconds() / 60))
                 freq_str = f"{int(median_diff.total_seconds() / 60)} Minutes"
            elif median_diff <= pd.Timedelta(minutes=30):
                 periods_per_year = 365 * 24 * (60 / (median_diff.total_seconds() / 60))
                 freq_str = f"{int(median_diff.total_seconds() / 60)} Minutes"
            elif median_diff <= pd.Timedelta(hours=1):
                periods_per_year = 365 * 24
                freq_str = "Hourly"
            elif median_diff <= pd.Timedelta(hours=4):
                 periods_per_year = 365 * (24 / (median_diff.total_seconds() / 3600))
                 freq_str = f"{int(median_diff.total_seconds() / 3600)} Hours"
            elif median_diff <= pd.Timedelta(days=1):
                periods_per_year = 365 # 或者使用 252 (交易日)
                freq_str = "Daily"
            else:
                # 對於週線或更長周期，年化意義不大，設為1
                periods_per_year = 1
                freq_str = f"Weekly or longer ({median_diff})"
                warnings.warn(f"Detected frequency {median_diff} is weekly or longer. Annualization might not be meaningful.")
        else:
             warnings.warn("Could not calculate time differences. Using default periods_per_year.")

    except Exception as e:
        warnings.warn(f"Error processing timestamp column: {e}. Using default periods_per_year.")
else:
    warnings.warn("Timestamp column not found. Using default periods_per_year=252.")

print(f"Detected data frequency: {freq_str} (Using {periods_per_year:.0f} periods per year for annualization)")


# ========== 初始化環境 ==========
env = TradingEnv(test_data, asset_name=ASSET_NAME)

# ========== 初始化Agent ==========
# --- 修正：_get_state 不需要額外參數 ---
# state_dim = env._get_state(env.current_step).shape[0] # 舊方式
state_dim = env._get_state().shape[0] # _get_state 會從 self 獲取 current_step
# ------------------------------------
action_dim = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 修正：初始化 Agent 時傳遞 state_dim ---
agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, device=device)
# -----------------------------------------

# --- 修正：使用 agent.load_model ---
# agent.policy_net.load_state_dict(torch.load(MODEL_PATH)) # 舊方式
agent.load_model(MODEL_PATH)
# ---------------------------------
agent.policy_net.eval() # 確保在評估模式
agent.epsilon = 0.0 # 評估時不探索

# ========== 測試並記錄 ==========
state = env.reset()
done = False

portfolio_values = [env.total_asset]
# returns = [] # 不再需要

while not done:
    # --- 修正：使用 agent.select_action ---
    # state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device) # 舊方式
    # with torch.no_grad():
    #     q_values = agent.policy_net(state_tensor)
    #     action = q_values.argmax().item()
    action = agent.select_action(state, evaluation=True)
    # ------------------------------------

    next_state, reward, done, info = env.step(action)
    portfolio_values.append(env.total_asset)
    # returns.append(reward) # 不再需要
    state = next_state

# ========== 資產曲線 ==========
plt.figure(figsize=(12,6))
plt.plot(portfolio_values, label='Portfolio Value')
plt.title('Portfolio Value Over Time')
plt.xlabel('Step')
plt.ylabel('Asset Value (USD)')
plt.legend()
plt.grid()
plt.show()

# ========== 計算績效指標 ==========

initial_asset = portfolio_values[0]
final_asset = portfolio_values[-1]
total_return = (final_asset - initial_asset) / initial_asset * 100

# 最大回撤
peak = portfolio_values[0]
max_drawdown = 0
for value in portfolio_values:
    if value > peak:
        peak = value
    drawdown = (peak - value) / peak
    if drawdown > max_drawdown:
        max_drawdown = drawdown

# === 修正：計算真實的週期回報率 ===
portfolio_array = np.array(portfolio_values)
# 計算每個週期的百分比變化率
# portfolio_array[1:] 是從第二個時間點開始的所有資產值
# portfolio_array[:-1] 是從第一個時間點開始到倒數第二個的所有資產值
# (portfolio_array[1:] - portfolio_array[:-1]) / portfolio_array[:-1] 就是每個週期的回報率
# 需要處理 portfolio_array[:-1] 中可能為零的情況
safe_denominators = np.where(portfolio_array[:-1] == 0, 1e-8, portfolio_array[:-1]) # 避免除以零
period_returns = (portfolio_array[1:] - portfolio_array[:-1]) / safe_denominators

# === 使用修正後的 period_returns 計算年化指標 ===
# 使用動態計算的 periods_per_year 進行年化
# 確保 period_returns 不是空的
if len(period_returns) > 0:
    # --- 新增：設定無風險利率 (年化) ---
    risk_free_rate = 3.0  # 假設年化無風險利率為 0%
    # ------------------------------------

    annualized_return = np.mean(period_returns) * periods_per_year
    annualized_volatility = np.std(period_returns) * np.sqrt(periods_per_year)

    # --- 修正：夏普比率計算，減去無風險利率 ---
    # 計算超額報酬
    excess_return = annualized_return - risk_free_rate
    # 計算夏普比率
    sharpe_ratio = excess_return / (annualized_volatility + 1e-8) # 加上 epsilon 避免除以零
    # ------------------------------------------
else:
    # 如果沒有週期回報（例如只有一步），則無法計算年化指標
    annualized_return = 0.0
    annualized_volatility = 0.0
    sharpe_ratio = 0.0 # 無法計算夏普比率


# ========== 顯式列印所有指標 ==========
print("="*60)
print(f"資產初始值 Initial Asset:    ${initial_asset:.2f}")
print(f"資產最終值 Final Asset:      ${final_asset:.2f}")
print(f"總報酬率 Total Return:       {total_return:.2f}%")
print(f"最大回撤 Max Drawdown:       {max_drawdown:.2%}")
print(f"年化報酬率 Annualized Return: {annualized_return:.2%}")
print(f"年化波動率 Annualized Vol:    {annualized_volatility:.2%}")
print(f"夏普比率 Sharpe Ratio:        {sharpe_ratio:.2f}")
print("="*60)
