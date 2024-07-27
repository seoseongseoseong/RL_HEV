import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.td3.policies import TD3Policy, ContinuousCritic
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import polyak_update
from typing import Optional, List, Type

class CustomActor(nn.Module):
    def __init__(self, observation_space, action_space, net_arch, activation_fn=nn.ReLU):
        super(CustomActor, self).__init__()
        self.feature_extractor = self.create_mlp(observation_space.shape[0], net_arch[-1], net_arch[:-1], activation_fn)
        self.mu = nn.Linear(net_arch[-1], action_space.shape[0])
        self.log_std = nn.Parameter(torch.zeros(1))  # std를 하나의 값으로 설정

    def forward(self, obs):
        features = self.feature_extractor(obs)
        mu = self.mu(features)
        return mu

    def get_log_std(self):
        return self.log_std

    def create_mlp(self, input_dim, output_dim, net_arch, activation_fn):
        layers = []
        last_dim = input_dim
        for layer_size in net_arch:
            layers.append(nn.Linear(last_dim, layer_size))
            layers.append(activation_fn())
            last_dim = layer_size
        layers.append(nn.Linear(last_dim, output_dim))
        return nn.Sequential(*layers)

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)

class CustomCritic(nn.Module):
    def __init__(self, observation_space, action_space, net_arch, activation_fn=nn.ReLU):
        super(CustomCritic, self).__init__()
        self.q1 = self.create_mlp(observation_space.shape[0] + action_space.shape[0], 1, net_arch, activation_fn)
        self.q2 = self.create_mlp(observation_space.shape[0] + action_space.shape[0], 1, net_arch, activation_fn)

    def forward(self, obs, action):
        q1 = self.q1(torch.cat([obs, action], dim=1))
        q2 = self.q2(torch.cat([obs, action], dim=1))
        return q1, q2

    def q1_forward(self, obs, action):
        return self.q1(torch.cat([obs, action], dim=1))

    def create_mlp(self, input_dim, output_dim, net_arch, activation_fn):
        layers = []
        last_dim = input_dim
        for layer_size in net_arch:
            layers.append(nn.Linear(last_dim, layer_size))
            layers.append(activation_fn())
            last_dim = layer_size
        layers.append(nn.Linear(last_dim, output_dim))
        return nn.Sequential(*layers)

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)

class CustomTD3Policy(TD3Policy):
    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = CustomActor(self.observation_space, self.action_space, self.net_arch, self.activation_fn)
        self.actor_target = CustomActor(self.observation_space, self.action_space, self.net_arch, self.activation_fn)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = CustomCritic(self.observation_space, self.action_space, self.net_arch, self.activation_fn)
        self.critic_target = CustomCritic(self.observation_space, self.action_space, self.net_arch, self.activation_fn)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_schedule(1))
        self.critic.optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_schedule(1))

    def _predict(self, observation, deterministic: bool = False) -> torch.Tensor:
        # Get the deterministic action (mu)
        mean_actions = self.actor(observation)
        if deterministic:
            return mean_actions
        # Add Gaussian noise
        log_std = self.actor.get_log_std()
        std = torch.exp(log_std)
        return torch.normal(mean_actions, std.expand_as(mean_actions))

    def train(self, gradient_steps: int, batch_size: int) -> None:
        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # Compute the target Q value
            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (torch.randn_like(replay_data.actions) * self.target_policy_noise).clamp(
                    -self.target_noise_clip, self.target_noise_clip
                )
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the target Q value
                target_q1, target_q2 = self.critic_target(replay_data.next_observations, next_actions)
                target_q = torch.min(target_q1, target_q2)
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * target_q

            # Get current Q estimates
            current_q1, current_q2 = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if gradient_step % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(replay_data.observations)).mean()

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                # Update the frozen target models
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
