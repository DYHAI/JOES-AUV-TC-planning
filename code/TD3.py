import torch as T
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from networks import ActorNetwork, CriticNetwork, StatePredictor
from buffer import ReplayBuffer
from optimum_degree_memory import Replay_optimum_degree  # 新增

device = T.device("cpu")  # 强制使用CPU


class TD3:
    def __init__(self, agent_num, alpha, beta, state_dim0, state_dim, action_dim, actor_fc1_dim, actor_fc2_dim,
                 critic_fc1_dim, critic_fc2_dim, ckpt_dir, gamma=0.99, tau=0.005, action_noise=0.1,
                 policy_noise=0.2, policy_noise_clip=0.5, delay_time=2, max_size=1000000,
                 batch_size=256, op_batch_size=128):
        # 多智能体情况添加编号
        self.agent_num = agent_num
        # alpha是actor网络更新的学习率，beta是critic的
        self.gamma = gamma                          # discout系数
        self.tau = tau                              # Interpolation factor in polyak averaging for target networks
        self.action_noise = action_noise
        self.policy_noise = policy_noise
        self.policy_noise_clip = policy_noise_clip
        self.delay_time = delay_time
        self.update_time = 0
        self.checkpoint_dir = ckpt_dir

        self.actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                   fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        self.critic1 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.critic2 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                     fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)

        self.target_actor = ActorNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                         fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        self.target_critic1 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)
        self.target_critic2 = CriticNetwork(beta=beta, state_dim=state_dim, action_dim=action_dim,
                                            fc1_dim=critic_fc1_dim, fc2_dim=critic_fc2_dim)

        self.memory = ReplayBuffer(max_size=max_size, state_dim=state_dim, action_dim=action_dim,
                                   batch_size=batch_size)

        self.update_network_parameters(tau=1.0)

        self.statePredictor = StatePredictor(alpha=alpha, state_dim0=state_dim0, action_dim=action_dim,
                                             fc1_dim=actor_fc1_dim, fc2_dim=actor_fc2_dim)
        # self.network_RNN = network_RNN(alpha=alpha,rnn_hidden_dim=actor_fc1_dim,state_dim0=state_dim0, action_dim=action_dim)

        # self.position_learn = True  # 设置 position_learn 属性为 True

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for actor_params, target_actor_params in zip(self.actor.parameters(),
                                                     self.target_actor.parameters()):
            target_actor_params.data.copy_(tau * actor_params + (1 - tau) * target_actor_params)

        for critic1_params, target_critic1_params in zip(self.critic1.parameters(),
                                                         self.target_critic1.parameters()):
            target_critic1_params.data.copy_(tau * critic1_params + (1 - tau) * target_critic1_params)

        for critic2_params, target_critic2_params in zip(self.critic2.parameters(),
                                                         self.target_critic2.parameters()):
            target_critic2_params.data.copy_(tau * critic2_params + (1 - tau) * target_critic2_params)

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation, train=True):
        # 调整成evaluation模式
        self.actor.eval()
        state = T.tensor(np.array([observation]), dtype=T.float, device=device)  # 在CPU上创建张量
        # 更新动作(策略网络——直接通过网络输出a)
        action = self.actor.forward(state)

        if train:
            # exploration noise
            noise = T.tensor(np.random.normal(loc=0.0, scale=self.action_noise),
                             dtype=T.float, device=device)  # 在CPU上创建噪声张量
            action = T.clamp(action+noise, -1, 1)
        self.actor.train()

        return action.squeeze().detach().cpu().numpy()
    ######## 状态更新网络
    def position_learn(self, current_states, actions, next_states):
        next_states = next_states.float()
        current_states = current_states.float()
        actions = actions.float()
        criterion = nn.MSELoss()  # 使用均方误差作为损失函数
        inputs = T.cat((current_states, actions), -1)  # 将状态和动作合并为输入
        inputs = inputs.type(T.float32)

        outputs = self.statePredictor.forward(inputs)

        statePredictor_loss = criterion(outputs, next_states)
        # 反向传播和优化器步骤
        self.statePredictor.optimizer.zero_grad()
        statePredictor_loss = statePredictor_loss.float()
        statePredictor_loss.backward()
        self.statePredictor.optimizer.step()

    def position_choose(self, current_states, actions):
        current_states = current_states.float()
        inputs = T.cat((current_states, actions), -1)  # 将状态和动作合并为输入
        inputs = inputs.type(T.float32)

        outputs = self.statePredictor.forward(inputs)

        next_position = outputs
        return next_position.squeeze().detach().cpu().numpy()

    def learn(self):
        if not self.memory.ready():
            return

        states, actions, rewards, states_, terminals = self.memory.sample_buffer()
        states_tensor = T.tensor(states, dtype=T.float, device=device)  # 在CPU上创建张量
        actions_tensor = T.tensor(actions, dtype=T.float, device=device)
        rewards_tensor = T.tensor(rewards, dtype=T.float, device=device)
        next_states_tensor = T.tensor(states_, dtype=T.float, device=device)
        terminals_tensor = T.tensor(terminals, device=device)

        with T.no_grad():
            next_actions_tensor = self.target_actor.forward(next_states_tensor)
            action_noise = T.tensor(np.random.normal(loc=0.0, scale=self.policy_noise),
                                    dtype=T.float, device=device)  # 在CPU上创建噪声张量
            # smooth noise
            action_noise = T.clamp(action_noise, -self.policy_noise_clip, self.policy_noise_clip)
            next_actions_tensor = T.clamp(next_actions_tensor+action_noise, -1, 1)
            q1_ = self.target_critic1.forward(next_states_tensor, next_actions_tensor).view(-1)
            q2_ = self.target_critic2.forward(next_states_tensor, next_actions_tensor).view(-1)
            q1_[terminals_tensor] = -100
            q2_[terminals_tensor] = -100
            critic_val = T.min(q1_, q2_)
            target = rewards_tensor + self.gamma * critic_val
        q1 = self.critic1.forward(states_tensor, actions_tensor).view(-1)
        q2 = self.critic2.forward(states_tensor, actions_tensor).view(-1)

        critic1_loss = F.mse_loss(q1, target.detach())
        critic2_loss = F.mse_loss(q2, target.detach())
        critic_loss = critic1_loss + critic2_loss
        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.update_time += 1
        if self.update_time % self.delay_time != 0:
            return

        new_actions_tensor = self.actor.forward(states_tensor)
        q1 = self.critic1.forward(states_tensor, new_actions_tensor)
        actor_loss = -T.mean(q1)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def save_models(self, episode):
        self.actor.save_checkpoint(self.checkpoint_dir + 'Actor/TD3_actor_{}.pth'.format(episode))
        self.target_actor.save_checkpoint(self.checkpoint_dir +
                                          'Target_actor/TD3_target_actor_{}.pth'.format(episode))
        self.critic1.save_checkpoint(self.checkpoint_dir + 'Critic1/TD3_critic1_{}.pth'.format(episode))
        self.target_critic1.save_checkpoint(self.checkpoint_dir +
                                            'Target_critic1/TD3_target_critic1_{}.pth'.format(episode))
        self.critic2.save_checkpoint(self.checkpoint_dir + 'Critic2/TD3_critic2_{}.pth'.format(episode))
        self.target_critic2.save_checkpoint(self.checkpoint_dir +
                                            'Target_critic2/TD3_target_critic2_{}.pth'.format(episode))

    def load_models(self, episode):
        self.actor.load_checkpoint(self.checkpoint_dir + 'Actor/TD3_actor_{}.pth'.format(episode))
        self.target_actor.load_checkpoint(self.checkpoint_dir +
                                          'Target_actor/TD3_target_actor_{}.pth'.format(episode))
        self.critic1.load_checkpoint(self.checkpoint_dir + 'Critic1/TD3_critic1_{}.pth'.format(episode))
        self.target_critic1.load_checkpoint(self.checkpoint_dir +
                                            'Target_critic1/TD3_target_critic1_{}.pth'.format(episode))
        self.critic2.load_checkpoint(self.checkpoint_dir + 'Critic2/TD3_critic2_{}.pth'.format(episode))
        self.target_critic2.load_checkpoint(self.checkpoint_dir +
                                            'Target_critic2/TD3_target_critic2_{}.pth'.format(episode))

