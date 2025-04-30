# replay_buffer.py (Simplified)
# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import numpy as np
from typing import Dict, Union


class ReplayBuffer:
    """Replay buffer for off-policy algorithms like SAC."""

    def __init__(
            self,
            buffer_size: int,
            obs_dim: int,
            action_dim: int,
            device: Union[torch.device, str] = "cuda:0"
        ):
        """Initialize a replay buffer.
        
        Args:
            buffer_size: Maximum number of transitions to store.
            obs_dim: Dimension of the observation space.
            action_dim: Dimension of the action space.
            device: Device to store the tensors on.
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        
        # Buffers for storing transitions
        self.observations = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self.next_observations = torch.zeros((buffer_size, obs_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        
        # Pointer to current position in buffer
        self.idx = 0
        # Current size of buffer (may be less than buffer_size initially)
        self.current_size = 0
    
    def add(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
            reward: torch.Tensor,
            next_obs: torch.Tensor,
            done: torch.Tensor
        ) -> None:
        """Add a transition to the buffer.
        
        Args:
            obs: Observation tensor.
            action: Action tensor.
            reward: Reward tensor.
            next_obs: Next observation tensor.
            done: Done tensor.
        """
        # Make sure all inputs are tensors on the correct device
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)
        
        # Ensure proper shape for reward and done
        if reward.dim() == 0:
            reward = reward.unsqueeze(0)
        if done.dim() == 0:
            done = done.unsqueeze(0)
            
        # Store transition
        self.observations[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward.reshape(-1)
        self.next_observations[self.idx] = next_obs
        self.dones[self.idx] = done.reshape(-1)
        
        # Update pointer and size
        self.idx = (self.idx + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample.
            
        Returns:
            Dictionary containing the sampled transitions.
        """
        # Make sure we have enough samples
        batch_size = min(batch_size, self.current_size)
        
        # Sample indices
        indices = torch.randint(0, self.current_size, (batch_size,), device=self.device)
        
        return {
            "obs": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_observations[indices],
            "dones": self.dones[indices]
        }
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if the buffer has enough samples for a batch.
        
        Args:
            batch_size: Size of the batch to check for.
            
        Returns:
            Whether the buffer has enough samples.
        """
        return self.current_size >= batch_size
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.idx = 0
        self.current_size = 0
        
        # Reset buffers
        self.observations.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.next_observations.zero_()
        self.dones.zero_()
    
    def to(self, device: Union[torch.device, str]) -> None:
        """Move the buffer to a specific device.
        
        Args:
            device: Device to move the buffer to.
        """
        self.device = device
        
        self.observations = self.observations.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.next_observations = self.next_observations.to(device)
        self.dones = self.dones.to(device)