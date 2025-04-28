import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DQNTRADING.env.trading_env import TradingEnv
from agent.dqn_agent import DQNAgent

# ========== 設定 ==========
MODEL_PATH = r'D:\my_code\DQNTrading\checkpoints\dqn_checkpoint_episode_1000.pth'
DATA_PATH = r'D:\my_code\DQNTrading\Data\btc_1d_2020_2024.csv'
ASSET_NAME = 'BTC'

# ========== 載入資料 ==========
test_data = pd.read_csv(DATA_PATH)

# ========== 初始化環境 ==========
env = TradingEnv(test_data, asset_name=ASSET_NAME)

# ========== 初始化Agent ==========
state_dim = env._get_state().shape[0]
action_dim = 7
agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
agent.policy_net.load_state_dict(torch.load(MODEL_PATH))
agent.policy_net.eval()
agent.epsilon = 0.0

# ========== 測試並記錄 ==========
state = env.reset()
done = False

portfolio_values = [env.total_asset]
returns = []

while not done:
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        q_values = agent.policy_net(state_tensor)
        action = q_values.argmax().item()

    next_state, reward, done, info = env.step(action)
    portfolio_values.append(env.total_asset)
    returns.append(reward)
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

# 報酬率（Returns）
returns_array = np.array(returns)
daily_returns = returns_array  # 因為每小時一筆資料，直接視為小時報酬
annualized_return = np.mean(daily_returns) * 24 * 365
annualized_volatility = np.std(daily_returns) * np.sqrt(24 * 365)

sharpe_ratio = (annualized_return / (annualized_volatility + 1e-8))

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
