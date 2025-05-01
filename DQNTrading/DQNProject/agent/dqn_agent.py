# filepath: d:\my_code\DQNTrading\DQNProject\agent\dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
# 新增 import
from torch.cuda.amp import autocast

# 簡單的MLP
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    # 修改 __init__ 簽名以接受新參數
    def __init__(self, state_dim, action_dim, device, batch_size=256, gamma=0.95, lr=5e-5, #<--- gamma 從 0.99 改為 0.95
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, memory_size=100000,
                 target_update_freq=500, grad_clip_norm=1.0): # <-- 加入 target_update_freq 和 grad_clip_norm
        self.device = device
        self.state_dim = state_dim # <-- 新增: 儲存 state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.target_update_freq = target_update_freq # <-- 儲存參數
        self.grad_clip_norm = grad_clip_norm # <-- 儲存參數
        self.train_steps = 0 # 追蹤訓練步數

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # use Huber loss for stability

        # 初始化 GradScaler
        # 只有在 CUDA 可用且 device 是 cuda 時啟用 AMP
        self.amp_enabled = torch.cuda.is_available() and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp_enabled) if self.amp_enabled else None

    # select_action 方法 (加入 evaluation 參數)
    def select_action(self, state, evaluation=False): # <-- 新增 evaluation 參數
        """選擇動作，evaluation 模式下不使用 epsilon-greedy"""
        if not evaluation and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            # 確保 state 是 numpy array
            if not isinstance(state, np.ndarray):
                state = np.array(state)
            # 檢查 state 維度是否匹配 (需要 self.state_dim)
            if state.shape[0] != self.state_dim:
                 print(f"Warning: State dimension mismatch in select_action. Expected {self.state_dim}, got {state.shape}. Returning random action.")
                 return random.randrange(self.action_dim)

            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.policy_net.eval() # 確保在推論模式
            with torch.no_grad():
                 # 推論時通常不需要 autocast
                 q_values = self.policy_net(state)
            self.policy_net.train() # 恢復訓練模式
            return q_values.argmax().item()

    # push 方法 (確保儲存 numpy)
    def push(self, state, action, reward, next_state, done):
        # 確保儲存的是 numpy array
        if isinstance(state, torch.Tensor): state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor): next_state = next_state.cpu().numpy()
        if not isinstance(state, np.ndarray): state = np.array(state)
        if not isinstance(next_state, np.ndarray): next_state = np.array(next_state)

        # 檢查維度 (可選)
        # if state.shape[0] != self.state_dim or next_state.shape[0] != self.state_dim:
        #     print(f"Warning: Dimension mismatch when pushing to memory. State: {state.shape}, Next State: {next_state.shape}")
        #     return

        self.memory.append((state, action, reward, next_state, done))

    # sample 方法 (修正：加入 self)
    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # 批次轉換為 NumPy array 再轉 Tensor
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device) # 或者 BoolTensor

        return states, actions, rewards, next_states, dones

    # train_step 方法 (加入梯度裁剪)
    def train_step(self, grad_clip=None): # <--- 加入 grad_clip 參數
        if len(self.memory) < self.batch_size:
            return None # 返回 None 表示未訓練

        states, actions, rewards, next_states, dones = self.sample()

        self.policy_net.train() # 確保在訓練模式

        # 使用 autocast 包裹前向傳播和損失計算
        with autocast(enabled=self.amp_enabled):
            # 計算 Q(s, a)
            q_values = self.policy_net(states).gather(1, actions)

            # 計算 V(s') = max_a' Q_target(s', a')
            with torch.no_grad():
                # double DQN target
                next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
                next_q_values_target = self.target_net(next_states).gather(1, next_actions)

            # 計算目標 Q 值: r + gamma * V(s') * (1 - done)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values_target

            # 計算損失
            loss = self.criterion(q_values, target_q_values)

        # 反向傳播與優化
        self.optimizer.zero_grad()
        # 使用 scaler 進行梯度縮放 (如果啟用 AMP)
        if self.amp_enabled and self.scaler:
            self.scaler.scale(loss).backward()
            # --- 加入梯度裁剪 (在 scaler.step 之前 unscale) ---
            self.scaler.unscale_(self.optimizer)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), grad_clip)
            # ----------------------------------------------------
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else: # 不使用 AMP
            loss.backward()
            # --- 加入梯度裁剪 (不使用 scaler) ---
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), grad_clip)
            # ------------------------------------
            self.optimizer.step()


        self.train_steps += 1 # 增加訓練步數計數器

        # --- 定期更新目標網路 (使用 self.target_update_freq) ---
        if self.train_steps % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item() # 返回 loss 值供追蹤

    # update_epsilon 方法 (保持不變)
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # update_target_network 方法 (保持不變)
    def update_target_network(self):
        print(f"Updating target network at train step {self.train_steps}") # Log 更新
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # save_model 方法 (新增)
    def save_model(self, path):
        """儲存模型權重"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(), # 儲存 scaler 狀態
            'epsilon': self.epsilon,
            'train_steps': self.train_steps,
        }, path)
        print(f"Model saved to {path}")

    # load_model 方法 (修改以兼容舊格式)
    def load_model(self, path):
        """載入模型權重，兼容舊格式 (直接 state_dict) 和新格式 (字典)"""
        checkpoint = torch.load(path, map_location=self.device) # 確保載入到正確設備

        # 檢查 checkpoint 是否為字典且包含預期鍵值
        if isinstance(checkpoint, dict) and 'policy_net_state_dict' in checkpoint:
            # --- 新格式 ---
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            # 嘗試載入 target_net (如果存在)
            if 'target_net_state_dict' in checkpoint:
                self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            else: # 如果沒有 target_net，則從 policy_net 複製
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # 嘗試載入 optimizer (如果存在)
            if 'optimizer_state_dict' in checkpoint:
                 try:
                     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                 except ValueError as e:
                      print(f"Warning: Could not load optimizer state_dict: {e}. Optimizer state might be reset.")

            # 嘗試載入 scaler (如果存在且啟用 AMP)
            if 'scaler_state_dict' in checkpoint and self.amp_enabled and self.scaler is not None:
                 try:
                     self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                 except Exception as e:
                      print(f"Warning: Could not load scaler state_dict: {e}. Scaler state might be reset.")

            self.epsilon = checkpoint.get('epsilon', self.epsilon_end) # 載入 epsilon
            self.train_steps = checkpoint.get('train_steps', 0) # 載入訓練步數
            print(f"Model loaded from {path} (New format).")

        elif isinstance(checkpoint, dict) and not 'policy_net_state_dict' in checkpoint:
             # --- 可能是舊格式，直接是 state_dict ---
             try:
                 self.policy_net.load_state_dict(checkpoint)
                 # 舊格式沒有 target_net, optimizer 等信息，需要重置
                 self.target_net.load_state_dict(self.policy_net.state_dict())
                 # epsilon 和 train_steps 也無法恢復
                 self.epsilon = self.epsilon_end # 評估模式設為最小值
                 self.train_steps = 0
                 print(f"Model loaded from {path} (Old format - state_dict only). Target net reset.")
             except Exception as e:
                 print(f"Error loading state_dict directly: {e}")
                 raise RuntimeError(f"Could not load checkpoint from {path}. Unknown format.")
        else:
             # --- 未知格式 ---
             raise RuntimeError(f"Could not load checkpoint from {path}. Unknown format.")


        self.policy_net.to(self.device) # 再次確保模型在正確設備
        self.target_net.to(self.device)
        self.target_net.eval() # 確保 target net 在評估模式
        print(f"Model loaded from {path}")