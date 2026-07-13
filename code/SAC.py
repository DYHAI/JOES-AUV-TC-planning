import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from networks import ActorNetwork, CriticNetwork, StatePredictor # 复用你定义的网络基础类
from  buffer import ReplayBuffer  # 复用你的经验回放缓冲区类
device = T.device("cpu") # 强制使用CPU对齐

# -------------------------------------------------------------------
# SAC 专用 Actor 网络 (需输出 mu 和 std)
# -------------------------------------------------------------------
class ActorNetworkSAC(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(ActorNetworkSAC, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim) # 对齐 TD3 的 LayerNorm
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.mu = nn.Linear(fc2_dim, action_dim)
        self.std = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        mu = self.mu(x)
        std = F.softplus(self.std(x)) # 确保标准差为正
        return mu, std

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))

# -------------------------------------------------------------------
# SAC 算法主体 - 接口完全对齐 TD3.py
# -------------------------------------------------------------------
class SACContinuous:
    def __init__(self, agent_num, alpha, beta, state_dim0, state_dim, action_dim, 
                 actor_fc1_dim, actor_fc2_dim, critic_fc1_dim, critic_fc2_dim, 
                 ckpt_dir, gamma=0.99, tau=0.005, alpha_lr=0.0003, target_entropy=None, 
                 batch_size=256, max_size=1000000):
        
        self.agent_num = agent_num
        self.gamma = gamma
        self.tau = tau
        self.checkpoint_dir = ckpt_dir
        self.batch_size = batch_size
        
        self.memory = ReplayBuffer(max_size=max_size, state_dim=state_dim, 
                                   action_dim=action_dim, batch_size=batch_size)
        # 1. 初始化网络 (对齐 TD3 的架构)
        self.actor = ActorNetworkSAC(alpha, state_dim, action_dim, actor_fc1_dim, actor_fc2_dim)
        
        # SAC 使用双 Q 网络缓解高估
        self.critic1 = CriticNetwork(beta, state_dim, action_dim, critic_fc1_dim, critic_fc2_dim)
        self.critic2 = CriticNetwork(beta, state_dim, action_dim, critic_fc1_dim, critic_fc2_dim)
        
        self.target_critic1 = CriticNetwork(beta, state_dim, action_dim, critic_fc1_dim, critic_fc2_dim)
        self.target_critic2 = CriticNetwork(beta, state_dim, action_dim, critic_fc1_dim, critic_fc2_dim)
        
        # 2. 状态预测器 (完全对齐 TD3)
        self.statePredictor = StatePredictor(alpha, state_dim0, action_dim, actor_fc1_dim, actor_fc2_dim)
        
        # 3. 自动熵调节 (SAC 特有)
        if target_entropy is None:
            self.target_entropy = -action_dim # 启发式设定
        else:
            self.target_entropy = target_entropy
            
        self.log_alpha = T.tensor(np.log(0.01), dtype=T.float, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.update_network_parameters(tau=1.0)

    def choose_action(self, observation, train=True):
        self.actor.eval()
        state = T.tensor(np.array([observation]), dtype=T.float, device=device)
        mu, std = self.actor.forward(state)
        
        if train:
            dist = Normal(mu, std)
            action_sample = dist.rsample() # 重参数化采样
            action = T.tanh(action_sample)
        else:
            action = T.tanh(mu) # 评估时使用均值
            
        self.actor.train()
        return action.squeeze().detach().cpu().numpy()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        # SAC 只需要更新 Critic 的 Target
        for c1, tc1 in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            tc1.data.copy_(tau * c1.data + (1 - tau) * tc1.data)
        for c2, tc2 in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            tc2.data.copy_(tau * c2.data + (1 - tau) * tc2.data)

    def position_learn(self, current_states, actions, next_states):
        # 逻辑与 TD3 里的完全一致
        self.statePredictor.optimizer.zero_grad()
        inputs = T.cat((current_states.float(), actions.float()), -1)
        outputs = self.statePredictor.forward(inputs)
        loss = F.mse_loss(outputs, next_states.float())
        loss.backward()
        self.statePredictor.optimizer.step()
    
    def remember(self, state, action, reward, state_, done):
        """将转换过程存储到经验回放池中"""
        self.memory.store_transition(state, action, reward, state_, done)

    def learn(self):
        if not self.memory.ready():
            return

        states, actions, rewards, states_, terminals = self.memory.sample_buffer()
        
        states_tensor = T.tensor(states, dtype=T.float, device=device)
        actions_tensor = T.tensor(actions, dtype=T.float, device=device)
        rewards_tensor = T.tensor(rewards, dtype=T.float, device=device).view(-1, 1)
        next_states_tensor = T.tensor(states_, dtype=T.float, device=device)
        terminals_tensor = T.tensor(terminals, device=device).view(-1, 1)

        # 1. 更新 Critic
        with T.no_grad():
            mu_, std_ = self.actor(next_states_tensor)
            dist_ = Normal(mu_, std_)
            next_actions_sample = dist_.rsample()
            next_actions = T.tanh(next_actions_sample)
            
            # 计算 log_prob 用于熵奖励
            log_prob_ = dist_.log_prob(next_actions_sample) - T.log(1 - next_actions.pow(2) + 1e-6)
            log_prob_ = log_prob_.sum(dim=1, keepdim=True)
            
            q1_ = self.target_critic1.forward(next_states_tensor, next_actions)
            q2_ = self.target_critic2.forward(next_states_tensor, next_actions)
            min_q_ = T.min(q1_, q2_) - T.exp(self.log_alpha) * log_prob_
            target = rewards_tensor + self.gamma * min_q_ * (1 - terminals_tensor.float())

        q1 = self.critic1.forward(states_tensor, actions_tensor)
        q2 = self.critic2.forward(states_tensor, actions_tensor)
        
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        # 2. 更新 Actor
        mu, std = self.actor(states_tensor)
        dist = Normal(mu, std)
        action_sample = dist.rsample()
        action = T.tanh(action_sample)
        
        log_prob = dist.log_prob(action_sample) - T.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        q1_new = self.critic1.forward(states_tensor, action)
        q2_new = self.critic2.forward(states_tensor, action)
        min_q_new = T.min(q1_new, q2_new)
        
        actor_loss = (T.exp(self.log_alpha) * log_prob - min_q_new).mean()
        
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # 3. 更新 Alpha
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.update_network_parameters()

    def save_models(self, episode):
        # 路径结构与 TD3 完全对齐
        self.actor.save_checkpoint(self.checkpoint_dir + f'Actor/SAC_actor_{episode}.pth')
        self.critic1.save_checkpoint(self.checkpoint_dir + f'Critic1/SAC_critic1_{episode}.pth')
        self.critic2.save_checkpoint(self.checkpoint_dir + f'Critic2/SAC_critic2_{episode}.pth')

    def load_models(self, episode):
        self.actor.load_checkpoint(self.checkpoint_dir + f'Actor/SAC_actor_{episode}.pth')
        self.critic1.load_checkpoint(self.checkpoint_dir + f'Critic1/SAC_critic1_{episode}.pth')
        self.critic2.load_checkpoint(self.checkpoint_dir + f'Critic2/SAC_critic2_{episode}.pth')