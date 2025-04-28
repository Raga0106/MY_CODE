import pandas as pd
import torch
import numpy as np
import os
from env.trading_env import TradingEnv
from agent.rainbow_agent import RainbowAgent

# ========== 設定 ==========
DATA_PATH = r'D:\my_code\DQNTrading\Data\btc_1d_2020_2024.csv'  # 你的測試資料
ASSET_NAME = 'BTC'
MODEL_PATH = r'checkpoints_rainbow/rainbow_checkpoint_episode_1000.pth'  # 訓練好的模型
SEED = 42

# ========== 隨機種子 ==========
torch.manual_seed(SEED)
np.random.seed(SEED)

# ========== 裝置設定 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ========== 載入資料 ==========
data = pd.read_csv(DATA_PATH)

# ========== 初始化環境 ==========
env = TradingEnv(data, asset_name=ASSET_NAME)

# ========== 初始化 Rainbow Agent ==========
state_dim = env._get_state().shape[0]
action_dim = 7
agent = RainbowAgent(state_dim=state_dim, action_dim=action_dim, device=device)

# 載入已訓練好的模型
agent.policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
agent.policy_net.eval()
print(f"Loaded model from {MODEL_PATH}")

# ========== 測試 ==========
state = env.reset()
done = False
total_reward = 0
rewards = []

while not done:
    action = agent.select_action(state)
    next_state, reward, done, info = env.step(action)

    total_reward += reward
    rewards.append(reward)

    state = next_state

# ========== 結果計算 ==========

def calculate_sharpe(returns, risk_free_rate=0.0):
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate
    mean_excess_return = excess_returns.mean()
    std_excess_return = excess_returns.std()
    if std_excess_return == 0:
        return 0.0
    sharpe_ratio = mean_excess_return / std_excess_return * np.sqrt(252)  # 假設一年252個交易日
    return sharpe_ratio

avg_reward = np.mean(rewards)
sharpe_ratio = calculate_sharpe(rewards)

print("\n===== Evaluation Result =====")
print(f"Total Reward: {total_reward:.4f}")
print(f"Average Reward per Step: {avg_reward:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print("==============================")
