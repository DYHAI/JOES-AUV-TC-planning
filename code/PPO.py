import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import os

# 确保设备兼容，优先CPU（对齐你的配置）
device = T.device("cpu")

# -------------------------------------------------------------------
# 策略网络 (Actor) - 优化数值稳定性
# -------------------------------------------------------------------
class PolicyNetContinuous(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)  # 层归一化提升稳定性
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.mu = nn.Linear(fc2_dim, action_dim)
        self.log_std = nn.Parameter(T.zeros(action_dim))  # 用可学习参数替代std层，更稳定
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha, eps=1e-5)
        self.to(device)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        mu = T.tanh(self.mu(x))  # 先缩放到[-1,1]，后续根据环境调整
        log_std = T.clamp(self.log_std, min=-20, max=2)  # 限制log_std范围，避免标准差过小/过大
        std = T.exp(log_std)
        return mu, std

    def save_checkpoint(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        T.save(self.state_dict(), file_path, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, file_path):
        if os.path.exists(file_path):
            self.load_state_dict(T.load(file_path, map_location=device))
        else:
            print(f"警告：模型文件 {file_path} 不存在，加载失败")

# -------------------------------------------------------------------
# 价值网络 (Critic) - 优化梯度稳定性
# -------------------------------------------------------------------
class ValueNet(nn.Module):
    def __init__(self, beta, state_dim, fc1_dim, fc2_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.v = nn.Linear(fc2_dim, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=beta, eps=1e-5)
        self.to(device)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.v(x)

    def save_checkpoint(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        T.save(self.state_dict(), file_path, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, file_path):
        if os.path.exists(file_path):
            self.load_state_dict(T.load(file_path, map_location=device))
        else:
            print(f"警告：模型文件 {file_path} 不存在，加载失败")

# -------------------------------------------------------------------
# PPO 算法主体 - 修复核心逻辑
# -------------------------------------------------------------------
class PPOContinuous:
    def __init__(self, agent_num, alpha, beta, state_dim0, state_dim, action_dim, 
                 actor_fc1_dim, actor_fc2_dim, critic_fc1_dim, critic_fc2_dim, 
                 ckpt_dir, gamma=0.99, lmbda=0.95, epochs=10, eps=0.2, batch_size=128):
        
        # 基础参数
        self.agent_num = agent_num
        self.gamma = gamma          # 折扣因子
        self.lmbda = lmbda          # GAE系数
        self.epochs = epochs        # 每轮更新迭代次数
        self.eps = eps              # PPO裁剪系数
        self.batch_size = batch_size
        self.ckpt_dir = ckpt_dir

        # 轨迹缓冲区（按episode存储，避免跨episode污染）
        self.buffer = {
            'states': [], 'actions': [], 'rewards': [], 
            'next_states': [], 'dones': []
        }

        # 初始化网络
        self.actor = PolicyNetContinuous(alpha, state_dim, action_dim, actor_fc1_dim, actor_fc2_dim)
        self.critic = ValueNet(beta, state_dim, critic_fc1_dim, critic_fc2_dim)

        # 兼容你的StatePredictor（如果不需要可以删除）
        try:
            from networks import StatePredictor
            self.statePredictor = StatePredictor(alpha, state_dim0, action_dim, actor_fc1_dim, actor_fc2_dim)
        except ImportError:
            self.statePredictor = None
            print("提示：未找到StatePredictor类，跳过初始化")

        # 兼容层（解决step函数中memory.ready()调用）
        self.memory = self  # 直接指向自身，简化封装

    def ready(self):
        """判断是否可以开始学习（buffer数据量≥batch_size）"""
        return len(self.buffer['states']) >= self.batch_size

    def choose_action(self, observation, train=True):
        """
        选择动作：训练时带探索，测试时取均值
        """
        self.actor.eval()  # 评估模式，禁用dropout等
        state = T.tensor(np.array([observation]), dtype=T.float, device=device)
        
        with T.no_grad():
            mu, std = self.actor(state)
            if train:
                # 训练时：从正态分布采样动作（探索）
                action_dist = T.distributions.Normal(mu, std)
                action = action_dist.sample()
            else:
                # 测试时：直接用均值（无探索）
                action = mu
        
        self.actor.train()  # 切回训练模式
        # 动作缩放：将[-1,1]映射到你环境需要的范围（根据step函数调整）
        action = action.squeeze().detach().cpu().numpy()
        return action

    def remember(self, state, action, reward, state_, done):
        """存储单步轨迹数据"""
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['next_states'].append(state_)
        self.buffer['dones'].append(done)

    def compute_gae(self, rewards, values, next_values, dones):
        """
        修复的GAE计算：按episode分割，考虑终止状态
        """
        advantages = []
        advantage = 0.0
        
        # 倒序遍历计算优势值
        for t in reversed(range(len(rewards))):
            # TD误差：r + γ*V(s') - V(s)
            td_error = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            # GAE：加权累加TD误差
            advantage = td_error + self.gamma * self.lmbda * (1 - dones[t]) * advantage
            advantages.insert(0, advantage)
        
        # 计算目标值：优势值 + 价值估计
        returns = np.array(advantages) + values
        # 标准化优势值（提升训练稳定性）
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return advantages, returns

    def learn(self):
        # 优化1: 增加数据量检查，避免过小批量更新 (建议 batch_size 改为 512 或 1024)
        if len(self.buffer['states']) < self.batch_size:
            return

        # 优化2: 直接处理 Tensor 列表，避免 np.array() 转换造成的巨大开销
        # state 和 next_state 本身就是 Tensor，直接 stack 最快
        states = T.stack(self.buffer['states']).to(device)
        next_states = T.stack(self.buffer['next_states']).to(device)
        
        # actions 和 rewards 是 numpy array 或 float，使用原有方式或直接转换
        actions = T.tensor(np.array(self.buffer['actions']), dtype=T.float).to(device)
        rewards = np.array(self.buffer['rewards'])
        dones = np.array(self.buffer['dones'])

        # 2. 计算价值估计和下一状态价值估计
        with T.no_grad():
            values = self.critic(states).squeeze().cpu().numpy()
            next_values = self.critic(next_states).squeeze().cpu().numpy()

        # 3. 计算GAE优势值和目标回报
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)
        advantages = T.tensor(advantages, dtype=T.float).to(device).unsqueeze(1)
        returns = T.tensor(returns, dtype=T.float).to(device).unsqueeze(1)

        # 4. 计算旧策略的log_prob
        with T.no_grad():
            mu_old, std_old = self.actor(states)
            old_dist = T.distributions.Normal(mu_old, std_old)
            old_log_probs = old_dist.log_prob(actions).sum(dim=1, keepdim=True)

        # 5. 分批次采样训练
        # 使用 TensorDataset 和 DataLoader 会更快，但这里保持逻辑简单进行索引切片
        dataset_length = len(states)
        indices = np.arange(dataset_length)
        
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_length, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # 取批次数据 (直接在 GPU/Tensor 上索引，速度极快)
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                # 6. 计算新策略的log_prob
                mu, std = self.actor(batch_states)
                dist = T.distributions.Normal(mu, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=1, keepdim=True)

                # 7. PPO裁剪损失
                ratio = T.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = T.clamp(ratio, 1 - self.eps, 1 + self.eps) * batch_advantages
                actor_loss = -T.mean(T.min(surr1, surr2))

                # 8. 价值网络损失
                critic_loss = F.mse_loss(self.critic(batch_states), batch_returns)

                # 9. 更新
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor.optimizer.step()

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic.optimizer.step()

        # 11. 清空buffer
        self.clear_buffer()

    def clear_buffer(self):
        """清空轨迹缓冲区"""
        for key in self.buffer.keys():
            self.buffer[key] = []

    def save_models(self, episode):
        """保存模型（按episode命名）"""
        actor_path = f"{self.ckpt_dir}/Actor/PPO_actor_{self.agent_num}_episode_{episode}.pth"
        critic_path = f"{self.ckpt_dir}/Critic/PPO_critic_{self.agent_num}_episode_{episode}.pth"
        self.actor.save_checkpoint(actor_path)
        self.critic.save_checkpoint(critic_path)

    def load_models(self, episode):
        """加载模型（按episode命名）"""
        actor_path = f"{self.ckpt_dir}/Actor/PPO_actor_{self.agent_num}_episode_{episode}.pth"
        critic_path = f"{self.ckpt_dir}/Critic/PPO_critic_{self.agent_num}_episode_{episode}.pth"
        self.actor.load_checkpoint(actor_path)
        self.critic.load_checkpoint(critic_path)