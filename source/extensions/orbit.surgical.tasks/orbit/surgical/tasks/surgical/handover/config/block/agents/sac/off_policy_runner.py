# offpolicy_runner.py (Simplified)
# Copyright (c) 2024, The ORBIT-Surgical Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import time
import statistics
import torch
import numpy as np
from typing import Dict, List, Optional, Union
from torch.utils.tensorboard import SummaryWriter


class OffPolicyRunner:
    """Simplified Off-Policy Runner for SAC training."""

    def __init__(
            self,
            env,
            agent,
            buffer,
            buffer_size=1_000_000,
            batch_size=2048,
            num_steps_per_env=24,
            num_warmup_steps=5000,
            max_iterations=1500,
            save_interval=50,
            num_learning_epochs=8,
            log_dir="logs",
            device="cuda:0"
        ):
        """Initialize the runner.
        
        Args:
            env: Environment to train in
            agent: SAC agent
            buffer: Replay buffer
            buffer_size: Size of the replay buffer
            batch_size: Batch size for training updates
            num_steps_per_env: Number of steps per environment before update
            num_warmup_steps: Steps to collect with random actions before learning
            max_iterations: Maximum number of training iterations
            save_interval: Save model every N iterations
            num_learning_epochs: Number of learning epochs per iteration
            log_dir: Directory for logs
            device: Device to train on
        """
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_steps_per_env = num_steps_per_env
        self.num_warmup_steps = num_warmup_steps
        self.max_iterations = max_iterations
        self.save_interval = save_interval
        self.num_learning_epochs = num_learning_epochs
        self.log_dir = log_dir
        self.device = device
        
        # Environment info
        self.num_envs = env.num_envs
        
        # Create log directories
        self.checkpoint_dir = os.path.join(log_dir, "model_checkpoints")
        self.tb_dir = os.path.join(log_dir, "tensorboard")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=self.tb_dir)
        
        # Training tracking variables
        self.current_iteration = 0
        self.total_steps = 0
        self.start_time = time.time()
        
        print(f"OffPolicyRunner initialized with {self.num_envs} environments")
        print(f"Training for max {self.max_iterations} iterations")
        print(f"Buffer size: {self.buffer_size}, Batch size: {self.batch_size}")
        print(f"Warmup steps: {self.num_warmup_steps}")
    
    def learn(self):
        """Train the agent using SAC."""
        # Reset the environment
        obs = self.env.reset()
        
        # Warmup phase - collect random actions to fill buffer
        if self.current_iteration == 0 and self.buffer.current_size < self.num_warmup_steps:
            print(f"Starting warmup phase to collect {self.num_warmup_steps} transitions with random actions...")
            
            warmup_steps = 0
            while warmup_steps < self.num_warmup_steps:
                # Generate random actions between -1 and 1
                random_actions = torch.rand(
                    (self.num_envs, self.agent.action_dim), 
                    device=self.device
                ) * 2 - 1
                
                # Step environment
                next_obs, rewards, dones, infos = self.env.step(random_actions)
                
                # Store transitions in buffer
                for i in range(self.num_envs):
                    self.buffer.add(
                        obs[i], 
                        random_actions[i], 
                        rewards[i], 
                        next_obs[i], 
                        dones[i]
                    )
                
                # Update observation
                obs = next_obs
                
                # Update progress
                warmup_steps += self.num_envs
                if warmup_steps % 1000 == 0:
                    print(f"Collected {warmup_steps}/{self.num_warmup_steps} random transitions")
            
            print("Warmup phase completed")
        
        # Main training loop
        print(f"Starting training for {self.max_iterations} iterations...")
        
        # Initialize metrics
        episode_rewards = []
        episode_lengths = []
        current_episode_rewards = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        current_episode_lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Start main loop
        for it in range(self.current_iteration, self.max_iterations):
            # Initialize metrics for this iteration
            iteration_start_time = time.time()
            self.current_iteration = it
            
            # Collect experience
            for step in range(self.num_steps_per_env):
                # Get actions from policy
                with torch.no_grad():
                    actions = self.agent.act(obs)
                
                # Step environment
                next_obs, rewards, dones, infos = self.env.step(actions)
                
                # Update episode stats
                current_episode_rewards += rewards.flatten()
                current_episode_lengths += 1
                
                # Store transitions in buffer
                for i in range(self.num_envs):
                    self.buffer.add(
                        obs[i], 
                        actions[i], 
                        rewards[i], 
                        next_obs[i], 
                        dones[i]
                    )
                
                # Check for episode terminations
                for i in range(self.num_envs):
                    if dones[i]:
                        # Store episode stats
                        episode_rewards.append(current_episode_rewards[i].item())
                        episode_lengths.append(current_episode_lengths[i].item())
                        
                        # Log to tensorboard
                        self.writer.add_scalar("train/episode_reward", current_episode_rewards[i].item(), self.total_steps)
                        self.writer.add_scalar("train/episode_length", current_episode_lengths[i].item(), self.total_steps)
                        
                        # Reset episode stats
                        current_episode_rewards[i] = 0
                        current_episode_lengths[i] = 0
                
                # Update observation
                obs = next_obs
                
                # Increment total steps
                self.total_steps += self.num_envs
            
            # Perform SAC updates
            actor_losses = []
            critic_losses = []
            alphas = []
            
            for _ in range(self.num_learning_epochs):
                # Sample from buffer
                batch = self.buffer.sample(self.batch_size)
                
                # Update agent
                metrics = self.agent.update(batch)
                
                # Record metrics
                actor_losses.append(metrics["actor_loss"])
                critic_losses.append(metrics["critic_loss"])
                alphas.append(metrics["alpha"])
            
            # Calculate metrics for this iteration
            avg_actor_loss = statistics.mean(actor_losses)
            avg_critic_loss = statistics.mean(critic_losses)
            avg_alpha = statistics.mean(alphas)
            
            # Calculate FPS
            iteration_time = time.time() - iteration_start_time
            fps = int(self.num_steps_per_env * self.num_envs / iteration_time)
            
            # Log to tensorboard
            self.writer.add_scalar("train/actor_loss", avg_actor_loss, it)
            self.writer.add_scalar("train/critic_loss", avg_critic_loss, it)
            self.writer.add_scalar("train/alpha", avg_alpha, it)
            self.writer.add_scalar("train/fps", fps, it)
            
            # Calculate elapsed time
            elapsed_time = time.time() - self.start_time
            elapsed_hours = int(elapsed_time / 3600)
            elapsed_minutes = int((elapsed_time % 3600) / 60)
            elapsed_seconds = int(elapsed_time % 60)
            
            # Calculate average episode reward
            recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
            avg_episode_reward = statistics.mean(recent_rewards) if recent_rewards else 0.0
            
            # Print progress
            print(f"Iteration {it}/{self.max_iterations} | "
                  f"Avg Reward: {avg_episode_reward:.2f} | "
                  f"Actor Loss: {avg_actor_loss:.4f} | "
                  f"Critic Loss: {avg_critic_loss:.4f} | "
                  f"Alpha: {avg_alpha:.4f} | "
                  f"FPS: {fps} | "
                  f"Time: {elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}")
            
            # Save model checkpoint
            if (it + 1) % self.save_interval == 0 or it == self.max_iterations - 1:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"model_{it+1}.pt")
                self.save(checkpoint_path)
                print(f"Model checkpoint saved to {checkpoint_path}")
        
        # Final save
        final_path = os.path.join(self.checkpoint_dir, "model_final.pt")
        self.save(final_path)
        print(f"Final model saved to {final_path}")
        print("Training completed.")
    
    def save(self, path):
        """Save the agent's networks and training state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save agent
        self.agent.save(path)
        
        # Save metadata
        metadata = {
            "current_iteration": self.current_iteration,
            "total_steps": self.total_steps,
        }
        
        meta_path = path.replace(".pt", ".meta.pt")
        torch.save(metadata, meta_path)
    
    def load(self, path):
        """Load the agent's networks and training state."""
        # Load agent
        self.agent.load(path)
        
        # Load metadata
        meta_path = path.replace(".pt", ".meta.pt")
        if os.path.exists(meta_path):
            metadata = torch.load(meta_path, map_location=self.device)
            self.current_iteration = metadata.get("current_iteration", 0)
            self.total_steps = metadata.get("total_steps", 0)
            print(f"Loaded checkpoint from iteration {self.current_iteration}")
        else:
            print("No metadata found, starting from iteration 0")
    
    def close(self):
        """Close the runner."""
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.close()