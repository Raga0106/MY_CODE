import pandas as pd
import torch
import random
import numpy as np
import os
from env.trading_env import TradingEnv
from agent.rainbow_agent import RainbowAgent

# ========== 設定 ==========
DATA_PATH = r'D:\my_code\DQNTrading\Data\btc_1d_2020_2024.csv'  # 你的資料路徑
ASSET_NAME = 'BTC'
NUM_EPISODES = 1000
SEED = 42
TARGET_UPDATE_FREQ = 1000  # 每幾步更新 target network

# ========== 隨機種子 ==========
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

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

# ========== 訓練參數 ==========
update_every = 4
step_count = 0

# ========== 訓練迴圈 ==========
for episode in range(1, NUM_EPISODES + 1):
    state = env.reset()
    done = False
    total_reward = 0
    total_steps = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.push(state, action, reward, next_state, done)

        step_count += 1
        if step_count % update_every == 0:
            agent.train_step()

        if step_count % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
            print(f"Target network updated at step {step_count}")

        state = next_state
        total_reward += reward
        total_steps += 1

    # 每個 episode 結束後顯示統計
    print(f"[Episode {episode:04d}] Reward: {total_reward:.4f}, Steps: {total_steps}")

    # 每100集保存一次模型
    if episode % 100 == 0:
        if not os.path.exists('checkpoints_rainbow/'):
            os.makedirs('checkpoints_rainbow/')
        torch.save(agent.policy_net.state_dict(), f'checkpoints_rainbow/rainbow_checkpoint_episode_{episode}.pth')
        print(f"Checkpoint saved at Episode {episode}")

print("Training complete.")
