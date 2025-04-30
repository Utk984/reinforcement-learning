# sac_agent.py
# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Dict, List, Tuple, Union


class Actor(nn.Module):
    """Actor network for SAC."""

    def __init__(
            self, 
            obs_dim: int, 
            action_dim: int, 
            hidden_dims: List[int], 
            activation: str = "elu",
            init_log_std: float = 0.0,
            log_std_bounds: Tuple[float, float] = (-20, 2),
        ):
        """Initialize the actor network.
        
        Args:
            obs_dim: Dimension of the observation space.
            action_dim: Dimension of the action space.
            hidden_dims: Dimensions of hidden layers.
            activation: Activation function to use.
            init_log_std: Initial value for log standard deviation.
            log_std_bounds: Bounds for log standard deviation.
        """
        super().__init__()
        
        # Activation function
        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Create layers
        layers = []
        prev_dim = obs_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(self.activation)
            prev_dim = dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Mean and log_std for the Gaussian policy
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0)
        nn.init.xavier_uniform_(self.log_std_layer.weight, gain=0.01)
        nn.init.constant_(self.log_std_layer.bias, init_log_std)
        
        # Log std bounds for numerical stability
        self.log_std_min, self.log_std_max = log_std_bounds
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass to get mean and log_std.
        
        Args:
            obs: Observation tensor.
            
        Returns:
            Tuple containing:
            - mean: Mean of the action distribution.
            - log_std: Log standard deviation of the action distribution.
        """
        shared_features = self.shared_layers(obs)
        
        mean = self.mean_layer(shared_features)
        log_std = self.log_std_layer(shared_features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample_actions(
            self, 
            obs: torch.Tensor, 
            deterministic: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions from the policy.
        
        Args:
            obs: Observation tensor.
            deterministic: Whether to sample deterministically.
            
        Returns:
            Tuple containing:
            - actions: Sampled actions.
            - log_probs: Log probabilities of the sampled actions.
            - mean: Mean of the action distribution.
        """
        mean, log_std = self.forward(obs)
        
        if deterministic:
            # Return the mean of the distribution
            return torch.tanh(mean), torch.zeros_like(mean), mean
        
        # Sample from the distribution
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        
        # Sample using reparameterization trick
        x = normal.rsample()
        
        # Apply tanh to bound actions between -1 and 1
        y = torch.tanh(x)
        
        # Calculate log probability accounting for the tanh transformation
        # log P(a|s) = log P(z|s) - sum log(1 - tanh(z)^2)
        log_prob = normal.log_prob(x) - torch.log(1 - y.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return y, log_prob, mean


class Critic(nn.Module):
    """Critic network for SAC."""

    def __init__(
            self, 
            obs_dim: int, 
            action_dim: int, 
            hidden_dims: List[int], 
            activation: str = "elu"
        ):
        """Initialize the critic network.
        
        Args:
            obs_dim: Dimension of the observation space.
            action_dim: Dimension of the action space.
            hidden_dims: Dimensions of hidden layers.
            activation: Activation function to use.
        """
        super().__init__()
        
        # Activation function
        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Create layers
        layers = []
        prev_dim = obs_dim + action_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(self.activation)
            prev_dim = dim
        
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # Initialize weights of the last layer
        nn.init.xavier_uniform_(self.output_layer.weight, gain=1.0)
        nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward pass to get Q-value.
        
        Args:
            obs: Observation tensor.
            actions: Action tensor.
            
        Returns:
            Q-value.
        """
        x = torch.cat([obs, actions], dim=-1)
        x = self.layers(x)
        q_value = self.output_layer(x)
        
        return q_value


class SACAgent:
    """Soft Actor-Critic agent."""

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            device: Union[torch.device, str],
            actor_hidden_dims: List[int] = [256, 256, 256],
            critic_hidden_dims: List[int] = [256, 256, 256],
            activation: str = "elu",
            init_log_std: float = 0.0,
            num_critics: int = 2,
            log_std_bounds: Tuple[float, float] = (-20, 2),
            gamma: float = 0.99,
            tau: float = 0.005,
            learning_rate: float = 3e-4,
            alpha_init: float = 0.2,
            auto_tune_alpha: bool = True,
            target_entropy_scale: float = 1.0,
            reward_scale: float = 1.0,
            clip_actions: bool = False,
            clip_observations: bool = False,
            use_grad_clip: bool = True,
            max_grad_norm: float = 1.0,
        ):
        """Initialize the SAC agent.
        
        Args:
            obs_dim: Dimension of the observation space.
            action_dim: Dimension of the action space.
            device: Device to use.
            actor_hidden_dims: Dimensions of hidden layers for actor.
            critic_hidden_dims: Dimensions of hidden layers for critic.
            activation: Activation function to use.
            init_log_std: Initial value for log standard deviation.
            num_critics: Number of critic networks (usually 2).
            log_std_bounds: Bounds for log standard deviation.
            gamma: Discount factor.
            tau: Soft update coefficient.
            learning_rate: Learning rate for optimization.
            alpha_init: Initial entropy coefficient.
            auto_tune_alpha: Whether to automatically tune alpha.
            target_entropy_scale: Scale for target entropy.
            reward_scale: Scale for rewards.
            clip_actions: Whether to clip actions.
            clip_observations: Whether to clip observations.
            use_grad_clip: Whether to use gradient clipping.
            max_grad_norm: Maximum gradient norm.
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.clip_actions = clip_actions
        self.clip_observations = clip_observations
        self.use_grad_clip = use_grad_clip
        self.max_grad_norm = max_grad_norm
        self.num_critics = num_critics
        self.action_dim = action_dim
        
        # Create actor network
        self.actor = Actor(
            obs_dim=obs_dim, 
            action_dim=action_dim,
            hidden_dims=actor_hidden_dims,
            activation=activation,
            init_log_std=init_log_std,
            log_std_bounds=log_std_bounds
        ).to(device)
        
        # Create critic networks (multiple critics for reduced overestimation bias)
        self.critics = nn.ModuleList()
        self.target_critics = nn.ModuleList()
        
        for _ in range(num_critics):
            critic = Critic(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dims=critic_hidden_dims,
                activation=activation
            ).to(device)
            
            target_critic = Critic(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dims=critic_hidden_dims,
                activation=activation
            ).to(device)
            
            # Initialize target critic with same weights as critic
            target_critic.load_state_dict(critic.state_dict())
            
            # Disable gradient updates for target network
            for param in target_critic.parameters():
                param.requires_grad = False
            
            self.critics.append(critic)
            self.target_critics.append(target_critic)
        
        # Setup optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critics_optimizers = [optim.Adam(critic.parameters(), lr=learning_rate) for critic in self.critics]
        
        # Entropy tuning
        self.auto_tune_alpha = auto_tune_alpha
        self.log_alpha = torch.tensor(np.log(alpha_init), requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()
        
        if auto_tune_alpha:
            # Target entropy: -dim(A) as default value
            self.target_entropy = -action_dim * target_entropy_scale
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
    
    def scale_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Scale observations if clipping is enabled.
        
        Args:
            obs: Observation tensor.
            
        Returns:
            Scaled observation tensor.
        """
        if self.clip_observations:
            # Clip to reasonable range (e.g., [-10, 10])
            return torch.clamp(obs, -10.0, 10.0)
        return obs
    
    def scale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Scale actions if clipping is enabled.
        
        Args:
            action: Action tensor.
            
        Returns:
            Scaled action tensor.
        """
        if self.clip_actions:
            # Already bounded by tanh, but can add additional scaling here if needed
            return torch.clamp(action, -1.0, 1.0)
        return action
    
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Select action based on observations.
        
        Args:
            obs: Observation tensor.
            deterministic: Whether to act deterministically.
            
        Returns:
            Action tensor.
        """
        with torch.no_grad():
            obs = self.scale_observation(obs)
            action, _, _ = self.actor.sample_actions(obs, deterministic=deterministic)
            action = self.scale_action(action)
        
        return action
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update the agent's networks based on a batch of experiences.
        
        Args:
            batch: Dictionary containing observation, action, reward, next_observation, and done.
            
        Returns:
            Dictionary of training metrics.
        """
        obs = self.scale_observation(batch["obs"])
        actions = batch["actions"]
        rewards = batch["rewards"] * self.reward_scale
        next_obs = self.scale_observation(batch["next_obs"])
        dones = batch["dones"]
        
        # Update critics
        next_actions, next_log_probs, _ = self.actor.sample_actions(next_obs)
        next_actions = self.scale_action(next_actions)
        
        next_q_values = []
        for target_critic in self.target_critics:
            next_q = target_critic(next_obs, next_actions)
            next_q_values.append(next_q)
        
        # Take the minimum over all critic networks
        next_q_values = torch.cat(next_q_values, dim=1)
        next_q_value = next_q_values.min(dim=1, keepdim=True)[0]
        
        # Compute entropy-regularized target Q-value
        target_q_value = rewards + (1 - dones) * self.gamma * (next_q_value - self.alpha * next_log_probs)
        target_q_value = target_q_value.detach()  # Don't backpropagate through target
        
        # Update each critic
        critics_losses = []
        for i, (critic, critic_optimizer) in enumerate(zip(self.critics, self.critics_optimizers)):
            current_q = critic(obs, actions)
            critic_loss = F.mse_loss(current_q, target_q_value)
            
            critic_optimizer.zero_grad()
            critic_loss.backward()
            
            if self.use_grad_clip:
                nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
            
            critic_optimizer.step()
            critics_losses.append(critic_loss.item())
        
        # Update actor and alpha
        new_actions, log_probs, _ = self.actor.sample_actions(obs)
        new_actions = self.scale_action(new_actions)
        
        # Compute Q-values of new actions using all critics
        q_values = []
        for critic in self.critics:
            q = critic(obs, new_actions)
            q_values.append(q)
        
        q_values = torch.cat(q_values, dim=1)
        min_q_value = q_values.min(dim=1, keepdim=True)[0]
        
        # Actor loss: maximize Q - alpha * log_prob
        actor_loss = (self.alpha.detach() * log_probs - min_q_value).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        
        if self.use_grad_clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        
        self.actor_optimizer.step()
        
        # Update alpha (automatic entropy tuning)
        alpha_loss = None
        if self.auto_tune_alpha:
            # Alpha loss: minimize -alpha * (log_prob + target_entropy)
            alpha_loss = -(self.log_alpha.exp() * (log_probs.detach() + self.target_entropy)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().detach()
        
        # Soft update of target critic networks
        for critic, target_critic in zip(self.critics, self.target_critics):
            for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Return metrics
        avg_q_value = min_q_value.mean().item()
        
        metrics = {
            "critic_loss": np.mean(critics_losses),
            "actor_loss": actor_loss.item(),
            "q_value": avg_q_value,
            "entropy": -log_probs.mean().item(),
            "alpha": self.alpha.item(),
        }
        if alpha_loss is not None:
            metrics["alpha_loss"] = alpha_loss.item()
        
        return metrics
    
    def save(self, path: str) -> None:
        """Save the agent's networks.
        
        Args:
            path: Directory path to save the networks.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state_dict = {
            "actor": self.actor.state_dict(),
            "log_alpha": self.log_alpha,
            "alpha": self.alpha,
        }
        
        # Save critic and target critic networks
        for i, (critic, target_critic) in enumerate(zip(self.critics, self.target_critics)):
            state_dict[f"critic{i}"] = critic.state_dict()
            state_dict[f"target_critic{i}"] = target_critic.state_dict()
        
        torch.save(state_dict, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load the agent's networks.
        
        Args:
            path: Path to load the networks from.
        """
        state_dict = torch.load(path, map_location=self.device)
        
        # Load actor
        self.actor.load_state_dict(state_dict["actor"])
        
        # Load critics and target critics
        for i in range(self.num_critics):
            if f"critic{i}" in state_dict:
                self.critics[i].load_state_dict(state_dict[f"critic{i}"])
            if f"target_critic{i}" in state_dict:
                self.target_critics[i].load_state_dict(state_dict[f"target_critic{i}"])
        
        # Load alpha
        if "log_alpha" in state_dict:
            self.log_alpha = state_dict["log_alpha"]
            self.alpha = state_dict["alpha"]
        
        print(f"Model loaded from {path}")