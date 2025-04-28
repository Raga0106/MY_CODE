import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from env.trading_env import TradingEnv
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

# ========== 測試並記錄交易 ==========
state = env.reset()
done = False

# 紀錄用
portfolio_values = [env.total_asset]
actions_taken = []  # 每步的動作
entry_prices = []   # 每次開倉的價格
exit_prices = []    # 每次平倉的價格
timestamps = []     # 每步時間（以index代替）

while not done:
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
    with torch.no_grad():
        q_values = agent.policy_net(state_tensor)
        action = q_values.argmax().item()

    next_state, reward, done, info = env.step(action)
    portfolio_values.append(env.total_asset)
    actions_taken.append(action)
    timestamps.append(env.current_step)

    # 記錄開倉和平倉價格
    if action == 1:  # 開多
        entry_prices.append((env.current_step, env.data.iloc[env.current_step]['close']))
    if action == 2:  # 開空
        entry_prices.append((env.current_step, env.data.iloc[env.current_step]['close']))
    if action in [3, 4]:  # 平多 or 平空
        exit_prices.append((env.current_step, env.data.iloc[env.current_step]['close']))

    state = next_state

# ========== 畫K棒與交易點 ==========
import mplfinance as mpf

# 把測試資料轉成mplfinance格式
ohlc_data = test_data[['open', 'high', 'low', 'close', 'volume']]
ohlc_data.index = pd.to_datetime(test_data.index, unit='s', origin='unix', errors='ignore') if not isinstance(test_data.index, pd.DatetimeIndex) else test_data.index

# 交易標記
buy_signals = [ (idx, price) for idx, price in entry_prices ]
sell_signals = [ (idx, price) for idx, price in exit_prices ]

# 把 index 轉成datetime方便對齊
if not isinstance(test_data.index, pd.DatetimeIndex):
    test_data.index = pd.to_datetime(test_data.index, unit='s', origin='unix', errors='ignore')

apds = []

# 加上買入箭頭
for idx, price in buy_signals:
    apds.append(mpf.make_addplot([np.nan if i != idx else price for i in range(len(test_data))], 
                                 type='scatter', markersize=100, marker='^', color='green'))

# 加上賣出箭頭
for idx, price in sell_signals:
    apds.append(mpf.make_addplot([np.nan if i != idx else price for i in range(len(test_data))], 
                                 type='scatter', markersize=100, marker='v', color='red'))

# 畫圖
mpf.plot(ohlc_data, type='candle', style='charles', addplot=apds, volume=True,
         title='Trading Behavior Visualization', figratio=(16,9), figscale=1.2)
