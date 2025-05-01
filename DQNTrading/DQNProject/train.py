import os, random, numpy as np, pandas as pd, torch
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime # <--- 新增 import

from env.trading_env_simple import TradingEnv
from agent.dqn_agent import DQNAgent

# ========== 超參數 ==========
DATA_PATH = r'D:\my_code\DQNTrading\Data\btc_1d_2020_2024.csv'
NUM_EPISODES = 1000
BATCH_SIZE = 256
TARGET_UPDATE_FREQUENCY = 500 # (修改) 目標網路更新頻率 (原 1000 -> 500)
UPDATE_EVERY = 4
EVAL_EVERY_EP = 100       # 每多少 episode 跑一次驗證
EARLY_STOP_PATIENCE = 5   # 驗證無進步達幾次就提早停止
GRAD_CLIP = 5.0           # 梯度裁剪

# ========== 隨機種子 & 裝置 ==========
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Logger ==========
# 建立一個包含時間戳記的唯一 log 目錄
log_dir = os.path.join("runs", "dqn_experiment_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir=log_dir) # <--- 使用新的 log_dir

# ========== 資料 & 環境 & Agent ==========
data = pd.read_csv(DATA_PATH)
train_env = TradingEnv(data)
# 用同一段資料做簡單驗證（也可另切驗證集）
val_env = TradingEnv(data)

state_dim = train_env.reset().shape[0]
agent = DQNAgent(state_dim, action_dim=3, device=device,
                 batch_size=BATCH_SIZE, target_update_freq=TARGET_UPDATE_FREQUENCY)

# ========== 訓練迴圈 ==========
step_count = 0
best_val_reward = -np.inf
no_improve_epochs = 0

for ep in trange(1, NUM_EPISODES+1, desc="Episodes"):
    state = train_env.reset()
    done = False
    ep_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = train_env.step(action)

        # --- 獎勵裁剪 ---
        reward = np.clip(reward, -1.0, 1.0) # <--- 新增：將獎勵限制在 [-1, 1]

        agent.push(state, action, reward, next_state, done)
        state = next_state
        ep_reward += reward
        step_count += 1

        # 訓練
        if len(agent.memory) >= BATCH_SIZE and step_count % UPDATE_EVERY == 0:
            loss = agent.train_step(grad_clip=GRAD_CLIP)
            writer.add_scalar("Loss/step", loss, step_count)

        # 更新 target network
        if step_count % TARGET_UPDATE_FREQUENCY == 0:
            agent.update_target_network()

    # Ep 結束後
    agent.update_epsilon()
    writer.add_scalar("Reward/train_episode", ep_reward, ep)
    writer.add_scalar("Epsilon/train", agent.epsilon, ep)

    # 隨機 sample 當前 state 的 Q‑value
    sample_q = agent.policy_net(torch.FloatTensor(state).unsqueeze(0).to(device))[0].cpu().detach().numpy()
    writer.add_scalars("Q_values", {"hold":sample_q[0], "long":sample_q[1], "short":sample_q[2]}, ep)

    # 定期驗證
    if ep % EVAL_EVERY_EP == 0:
        val_state = val_env.reset()
        val_done = False
        val_reward = 0
        while not val_done:
            a = agent.select_action(val_state, evaluation=True)
            val_state, r, val_done, _ = val_env.step(a)
            val_reward += r
        writer.add_scalar("Reward/val_episode", val_reward, ep)

        # Early stopping & best model
        if val_reward > best_val_reward:
            best_val_reward = val_reward
            no_improve_epochs = 0
            torch.save(agent.policy_net.state_dict(), "checkpoints/best_model.pth")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= EARLY_STOP_PATIENCE:
                print(f"No improvement for {EARLY_STOP_PATIENCE} evals. Early stopping.")
                break

    # 定期存檔
    if ep % 200 == 0:
        torch.save(agent.policy_net.state_dict(), f"checkpoints/dqn_ep{ep}.pth")

# 最終存檔
torch.save(agent.policy_net.state_dict(), "checkpoints/dqn_final.pth")
writer.close()
