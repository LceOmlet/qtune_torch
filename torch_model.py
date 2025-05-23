import sys
import datetime
import time
from collections import deque
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas
import heapq
import pymysql
import pymysql.cursors as pycursor

import gym
from gym import spaces
from gym.utils import seeding

import os
from copy import deepcopy
from configs import knob_config, config_dict

# Action normalization utility functions
def normalize_action(action, a_low, a_high):
    """
    Normalize actions to the range [-1, 1] for better training stability.
    Works with both PyTorch tensors and NumPy arrays.
    
    Args:
        action: The action tensor or numpy array to normalize
        a_low: Lower bound of action space
        a_high: Upper bound of action space
        
    Returns:
        Normalized action in range [-1, 1]
    """
    if isinstance(action, torch.Tensor):
        # Handle tensor case
        a_range = torch.tensor(a_high - a_low, device=action.device, dtype=action.dtype)
        a_middle = torch.tensor((a_high + a_low) / 2, device=action.device, dtype=action.dtype)
        
        # Map to [-1, 1]
        normalized = 2.0 * ((action - a_middle) / a_range)
        
        # Clip to ensure it's in range
        normalized = torch.clamp(normalized, -1.0, 1.0)
        return normalized
    else:
        # Handle numpy case
        a_range = a_high - a_low
        a_middle = (a_high + a_low) / 2
        
        # Map to [-1, 1]
        normalized = 2.0 * ((action - a_middle) / a_range)
        
        # Clip to ensure it's in range
        normalized = np.clip(normalized, -1.0, 1.0)
        return normalized

def denormalize_action(normalized_action, a_low, a_high):
    """
    Convert normalized actions from [-1, 1] back to the original action space.
    Works with both PyTorch tensors and NumPy arrays.
    
    Args:
        normalized_action: Action in normalized space [-1, 1]
        a_low: Lower bound of action space
        a_high: Upper bound of action space
        
    Returns:
        Action in original action space [a_low, a_high]
    """
    if isinstance(normalized_action, torch.Tensor):
        # Handle tensor case
        a_range = torch.tensor(a_high - a_low, device=normalized_action.device, dtype=normalized_action.dtype)
        a_middle = torch.tensor((a_high + a_low) / 2, device=normalized_action.device, dtype=normalized_action.dtype)
        
        # Map from [-1, 1] to [a_low, a_high]
        denormalized = (normalized_action * a_range / 2) + a_middle
        
        # Clip to ensure it's in range
        denormalized = torch.clamp(denormalized, a_low, a_high)
        return denormalized
    else:
        # Handle numpy case
        a_range = a_high - a_low
        a_middle = (a_high + a_low) / 2
        
        # Map from [-1, 1] to [a_low, a_high]
        denormalized = (normalized_action * a_range / 2) + a_middle
        
        # Clip to ensure it's in range
        denormalized = np.clip(denormalized, a_low, a_high)
        return denormalized

# Custom activation function for target range
def target_range(x, target_min=None, target_max=None):
    target_min = torch.tensor(target_min).to(x.device)
    target_max = torch.tensor(target_max).to(x.device)
    if target_min is None or target_max is None:
        # Default values if not provided
        target_min = 0
        target_max = 1
    x02 = torch.tanh(x) + 1  # x in range(0,2)
    scale = (target_max - target_min) / 2.
    return x02 * scale + target_min

class ZeroOrderOptimizer:
    def __init__(self, model, learning_rate, noise_std=None, noise_decay=None, lr_decay=None, decay_step=None, norm_rewards=None):
        self.model = model
        self._lr = learning_rate
        
        # Get zero_order parameters from config if available, otherwise use defaults
        try:
            zero_order_config = config_dict.get("zero_order", {})
            self._noise_std = noise_std if noise_std is not None else float(zero_order_config.get("noise_std", 0.1))
            self.noise_decay = noise_decay if noise_decay is not None else float(zero_order_config.get("noise_decay", 1.0))
            self.lr_decay = lr_decay if lr_decay is not None else float(zero_order_config.get("lr_decay", 1.0))
            self.decay_step = decay_step if decay_step is not None else int(zero_order_config.get("decay_step", 50))
            self.norm_rewards = norm_rewards if norm_rewards is not None else zero_order_config.get("norm_rewards", "true").lower() == "true"
        except Exception as e:
            # If any error occurs reading from config, use default values
            print(f"Error reading zero_order config, using defaults: {e}")
            self._noise_std = 0.1 if noise_std is None else noise_std
            self.noise_decay = 1.0 if noise_decay is None else noise_decay
            self.lr_decay = 1.0 if lr_decay is None else lr_decay
            self.decay_step = 50 if decay_step is None else decay_step
            self.norm_rewards = True if norm_rewards is None else norm_rewards
            
        self._population = None
        self._count = 0
        
        # Get the environment's action space bounds from the model
        # This assumes the model has access to the environment
        env = None
        if hasattr(model, 'env'):
            env = model.env
        elif hasattr(model, '_env'):
            env = model._env
        
        # Store environment for later use
        self.env = env

    @property
    def noise_std(self):
        step_decay = np.power(self.noise_decay, np.floor((1 + self._count) / self.decay_step))
        return self._noise_std * step_decay

    @property
    def lr(self):
        step_decay = np.power(self.lr_decay, np.floor((1 + self._count) / self.decay_step))
        return self._lr * step_decay

    def generate_population(self, npop=50):
        self._population = []
        # Store weights of the original model
        original_weights = [param.data.clone() for param in self.model.parameters()]
        
        for i in range(npop):
            # Create a new instance of the same model class
            new_model = type(self.model)(
                state_dim=self.model.state_dim,
                action_dim=self.model.action_dim,
                a_low=self.model.a_low,
                a_high=self.model.a_high
            )
            
            # Copy the weights from the original model
            for idx, param in enumerate(new_model.parameters()):
                param.data.copy_(original_weights[idx])
            
            # Apply noise to the weights
            new_model.E = []
            for param in new_model.parameters():
                noise = torch.randn_like(param)
                new_model.E.append(noise)
                param.data = param.data + self.noise_std * noise
            
            self._population.append(new_model)
        return self._population

    def update_population(self, rewards):
        if self._population is None:
            raise ValueError("populations is none, generate & eval it first")

        if self.norm_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        for i, param in enumerate(self.model.parameters()):
            w_updates = torch.zeros_like(param)
            for j, model in enumerate(self._population):
                w_updates = w_updates + (model.E[i] * rewards[j])
            param.data = param.data + (self.lr / (len(rewards) * self.noise_std)) * w_updates
        
        self._count = self._count + 1

    def get_model(self):
        return self.model

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, a_low, a_high):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.a_low = a_low
        self.a_high = a_high
        
        self.fc1 = nn.Linear(state_dim, 128)
        self.bn1 = nn.BatchNorm1d(128, affine=False)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, action_dim)
        
    def forward(self, state):
        # Check if we're in training mode and have a batch size of 1
        if self.training and state.size(0) == 1:
            # For batch size 1 during training, use a simple forward pass without batch norm
            x = F.relu(self.fc1(state))
            x = torch.tanh(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
        else:
            # Normal forward pass with batch norm
            x = F.relu(self.bn1(self.fc1(state)))
            x = torch.tanh(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
        
        # Apply custom target range activation
        return target_range(x, target_min=self.a_low, target_max=self.a_high)

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.state_fc = nn.Linear(state_dim, 128)
        self.action_fc = nn.Linear(action_dim, 128)
        
        self.fc1 = nn.Linear(256, 256)
        self.bn1 = nn.BatchNorm1d(256, affine=True)
        self.fc2 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.3)
        self.bn2 = nn.BatchNorm1d(64, affine=True)
        self.fc3 = nn.Linear(64, 1)
        
    def normalize_action(self, action, a_low=None, a_high=None):
        """
        Normalize actions to the range [-1, 1] for better training stability.
        
        Args:
            action: The action tensor to normalize
            a_low: Lower bound of action space (optional)
            a_high: Upper bound of action space (optional)
            
        Returns:
            Normalized action in range [-1, 1]
        """
        # If bounds are not provided, use default normalization
        if a_low is None or a_high is None:
            # Simple min-max normalization within the batch
            if len(action.shape) > 1 and action.shape[0] > 1:
                # For batches with multiple samples
                min_val = torch.min(action, dim=0)[0]
                max_val = torch.max(action, dim=0)[0]
                range_val = max_val - min_val
                # Avoid division by zero
                range_val = torch.where(range_val > 1e-6, range_val, torch.ones_like(range_val))
                return 2.0 * ((action - min_val) / range_val) - 1.0
            else:
                # For single samples, use a reasonable default range
                return torch.tanh(action)
        else:
            # Use the provided bounds for normalization
            a_range = torch.tensor(a_high - a_low, device=action.device, dtype=action.dtype)
            a_middle = torch.tensor((a_high + a_low) / 2, device=action.device, dtype=action.dtype)
            
            # Map to [-1, 1]
            normalized = 2.0 * ((action - a_middle) / a_range)
            
            # Clip to ensure it's in range
            normalized = torch.clamp(normalized, -1.0, 1.0)
            return normalized
        
    def forward(self, state, action):
        # Ensure both inputs are float32
        state = state.float()
        action = action.float()
        
        # Reshape state if needed (1D to 2D)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # Add batch dimension
            
        # Reshape action if needed (1D to 2D)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)  # Add batch dimension
            
        # Normalize the action input if model has access to environment bounds
        if hasattr(self, 'env'):
            action = self.normalize_action(action, self.env.a_low, self.env.a_high)
        else:
            # Use simple normalization if bounds aren't available
            action = self.normalize_action(action)
        
        state_out = self.state_fc(state)
        action_out = self.action_fc(action)
        state_out = torch.tanh(state_out)
        action_out = torch.tanh(action_out)
        # Merge state and action - concatenate instead of adding
        merged = torch.cat([state_out, action_out], dim=1)

        # Check if we're in training mode and have a batch size of 1
        if self.training and merged.size(0) == 1:
            # For batch size 1 during training, use a simple forward pass without batch norm
            x = self.fc1(merged)
            x = torch.tanh(x)
            x = torch.tanh(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
        else:
            # Normal forward pass with batch norm
            x = self.fc1(merged)
            x = self.bn1(x)
            x = torch.tanh(x)
            x = torch.tanh(self.fc2(x))
            x = self.dropout(x)
            x = self.bn2(x)
            x = self.fc3(x)
        
        return x

class ActorCritic:
    def __init__(self, env, learning_rate=0.001, train_min_size=32, size_mem=2000, size_predict_mem=2000, optimizer_type='adam'):
        self.env = env
        self.learning_rate = learning_rate
        self.train_min_size = train_min_size
        self.epsilon = 0.9
        self.epsilon_decay = 0.999
        self.gamma = 0.095
        self.tau = 0.125
        self.timestamp = int(time.time())
        self.optimizer_type = optimizer_type
        
        # Memory for experience replay
        self.memory = deque(maxlen=size_mem)
        self.mem_predicted = deque(maxlen=size_predict_mem)
        
        # Create actor and critic networks
        self.actor = Actor(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            a_low=env.a_low,
            a_high=env.a_high
        )
        self.target_actor = Actor(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            a_low=env.a_low,
            a_high=env.a_high
        )
        
        self.critic = Critic(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0]
        )
        self.target_critic = Critic(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0]
        )
        
        # Set environment reference for action normalization
        self.critic.env = env
        self.target_critic.env = env
        
        # Initialize target networks with the same weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Set up optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Initialize zero-order optimizer if selected
        if optimizer_type == 'zero_order':
            # Get zero_order parameters from config if available
            try:
                zero_order_config = config_dict.get("zero_order", {})
                noise_std = float(zero_order_config.get("noise_std", 0.1))
                noise_decay = float(zero_order_config.get("noise_decay", 0.99))
                lr_decay = float(zero_order_config.get("lr_decay", 0.99))
                decay_step = int(zero_order_config.get("decay_step", 50))
                norm_rewards = zero_order_config.get("norm_rewards", "true").lower() == "true"
                
                self.zero_order_opt = ZeroOrderOptimizer(
                    model=self.actor,
                    learning_rate=learning_rate,
                    noise_std=noise_std,
                    noise_decay=noise_decay,
                    lr_decay=lr_decay,
                    decay_step=decay_step,
                    norm_rewards=norm_rewards
                )
            except Exception as e:
                print(f"Error reading zero_order config, using defaults: {e}")
                self.zero_order_opt = ZeroOrderOptimizer(
                    model=self.actor,
                    learning_rate=learning_rate
                )
            # Set the environment attribute on the model for the optimizer to access
            self.actor.env = self.env

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_actor(self, samples, i):
        if self.optimizer_type == 'zero_order':
            # Generate population for zero-order optimization
            population = self.zero_order_opt.generate_population(npop=len(samples))
            rewards = []
            
            # Evaluate each model in population
            for model in population:
                cur_state, action, reward, new_state, _ = samples[0]  # Use first sample for evaluation
                # Normalize state
                if len(cur_state.shape) > 1 and cur_state.shape[0] > 0:
                    cur_state = (cur_state - min(cur_state[0]))/(max(cur_state[0])-min(cur_state[0]) + 1e-10)
                
                cur_state_tensor = torch.FloatTensor(cur_state).float()
                
                # Ensure tensor has batch dimension
                if len(cur_state_tensor.shape) == 1:
                    cur_state_tensor = cur_state_tensor.unsqueeze(0)
                
                # Set model to evaluation mode
                model.eval()
                with torch.no_grad():
                    predicted_action = model(cur_state_tensor).detach().numpy()
                # Set model back to training mode
                model.train()
                
                # Calculate critic gradients
                critic_state_tensor = torch.FloatTensor(cur_state).float()
                if len(critic_state_tensor.shape) == 1:
                    critic_state_tensor = critic_state_tensor.unsqueeze(0)
                    
                critic_action_tensor = torch.FloatTensor(predicted_action).float()
                if len(critic_action_tensor.shape) == 1:
                    critic_action_tensor = critic_action_tensor.unsqueeze(0)
                
                critic_action_tensor.requires_grad = True
                # Set critic to evaluation mode
                self.critic.eval()
                critic_value = self.critic(critic_state_tensor, critic_action_tensor)
                reward = critic_value.detach().cpu().numpy()
                # Set critic back to training mode
                self.critic.train()
                rewards.append(reward)
            
            # Update population using zero-order optimization
            self.zero_order_opt.update_population(np.array(rewards))
            self.actor = self.zero_order_opt.get_model()
        else:
            # Original Adam optimizer training
            for sample in samples:
                cur_state, action, reward, new_state, _ = sample
                
                # Normalize state
                if len(cur_state.shape) > 1 and cur_state.shape[0] > 0:
                    cur_state = (cur_state - min(cur_state[0]))/(max(cur_state[0])-min(cur_state[0]) + 1e-10)
                
                cur_state_tensor = torch.FloatTensor(cur_state).float()
                
                # Ensure tensor has batch dimension
                if len(cur_state_tensor.shape) == 1:
                    cur_state_tensor = cur_state_tensor.unsqueeze(0)
                
                # Get predicted action
                predicted_action = self.actor(cur_state_tensor)
                
                # Calculate critic gradients
                critic_state_tensor = torch.FloatTensor(cur_state).float()
                if len(critic_state_tensor.shape) == 1:
                    critic_state_tensor = critic_state_tensor.unsqueeze(0)
                
                critic_action_tensor = predicted_action.clone().detach()
                critic_action_tensor.requires_grad = True
                critic_value = self.critic(critic_state_tensor, critic_action_tensor)
                critic_value.backward()
                grads = critic_action_tensor.grad
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss = -torch.mean(predicted_action * grads)
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Log training information
                writer = open('training-results/training-' + str(self.timestamp), 'a')
                writer.write('grads')
                writer.write(f"{str(i)}\t{str(grads.detach().numpy().tolist())}\n")
                writer.write('cur_state\n')
                writer.write(str(cur_state)+'\n')
                writer.write('predicted_action\n')
                writer.write(str(predicted_action.detach().numpy())+'\n')
                writer.close()

    def _train_critic(self, samples, i):
        for sample in samples:
            cur_state, action, t_reward, new_state, done = sample
            reward = np.array([])
            reward = np.append(reward, t_reward[0])
            
            # Normalize state
            if len(cur_state.shape) > 1 and cur_state.shape[0] > 0:
                cur_state = (cur_state - min(cur_state[0]))/(max(cur_state[0])-min(cur_state[0]) + 1e-10)
            
            # Print debug info
            # print(f"Current state shape: {cur_state.shape}, Action shape: {action.shape}")
            
            # Convert to tensors with explicit float32 dtype
            cur_state_tensor = torch.FloatTensor(cur_state).float()
            action_tensor = torch.FloatTensor(action).float()
            new_state_tensor = torch.FloatTensor(new_state).float()
            
            # Make sure tensors have batch dimension
            if len(cur_state_tensor.shape) == 1:
                cur_state_tensor = cur_state_tensor.unsqueeze(0)
            if len(action_tensor.shape) == 1:
                action_tensor = action_tensor.unsqueeze(0)
            if len(new_state_tensor.shape) == 1:
                new_state_tensor = new_state_tensor.unsqueeze(0)
                
            print(f"Tensor shapes - State: {cur_state_tensor.shape}, Action: {action_tensor.shape}")
            
            # Get target action from target actor
            # Set target actor to evaluation mode
            self.target_actor.eval()
            with torch.no_grad():
                target_action = self.target_actor(new_state_tensor).detach()
            # Set target actor back to training mode
            self.target_actor.train()
            
            # Get future reward from target critic
            # Set target critic to evaluation mode
            self.target_critic.eval()
            with torch.no_grad():
                future_reward = self.target_critic(new_state_tensor, target_action)[0][0].detach().numpy()
            # Set target critic back to training mode
            self.target_critic.train()
            
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("future_reward:", future_reward)
            reward += self.gamma * future_reward
            print("reward:", reward)
            print("new_state_tensor:", new_state_tensor)
            print("target_action:", target_action.detach().numpy())
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_value = self.critic(cur_state_tensor, action_tensor)
            target_value = torch.FloatTensor(reward).float().unsqueeze(0).unsqueeze(0)  # Make sure it has [batch, 1] shape
            critic_loss = F.mse_loss(critic_value, target_value)
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Log training information
            writer = open('training-results/critic_training-' + str(self.timestamp), 'a')
            writer.write('epoch:\t'+str(i)+'\n')
            writer.write('critic_loss\t')
            writer.write(f"{str(critic_loss.item())}\n")
            writer.write('reward:\t')
            writer.write(f"{str(reward)}\n")
            writer.close()

    def train(self, i):
        self.batch_size = self.train_min_size
        if len(self.memory) < self.batch_size:
            return
            
        mem = list(self.memory)
        rewards = [i[2][0] for i in mem]
        indexs = heapq.nlargest(self.batch_size, range(len(rewards)), rewards.__getitem__)
        samples = []
        for i in indexs:
            samples.append(mem[i])
        samples = random.sample(list(self.memory), self.batch_size - 2)
        
        writer = open('training-results/training-' + str(self.timestamp), 'a')
        writer.write('samples\n')
        try:
            # Extract rewards manually instead of using numpy indexing
            rewards_str = [sample[2] for sample in samples]
            writer.write(f"{str(i)}\t{str(rewards_str)}\n")
        except Exception as e:
            print(f"Error writing samples: {e}")
            print(f"Samples: {samples}")
            raise e
        
        self._train_critic(samples, i)
        self._train_actor(samples, i)
        self.update_target()

    def update_target(self):
        # Soft update of target networks
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
            
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def get_calculate_knobs(self, action):
        caculate_knobs = list(knob_config)[len(action):]
        for k in caculate_knobs:
            if knob_config[k]['operator'] == 'multiply':
                pos_x = self.env.knob2pos[knob_config[k]['x']]
                pos_y = self.env.knob2pos[knob_config[k]['y']]
                tmp = action[pos_x] * action[pos_y]
                action = np.append(action, tmp)
        return action

    def act(self, cur_state):
        self.epsilon *= self.epsilon_decay
        action_tmp = None
        
        if np.random.random(1) < self.epsilon or len(self.memory) < self.batch_size:
            print("[Random Tuning]")
            action = np.round(self.env.action_space.sample())
            action = action.astype(np.float64)
            flag = 0
        else:
            print("[Model Tuning]")
            # Normalize state
            if len(cur_state.shape) > 1 and cur_state.shape[0] > 0:
                cur_state = (cur_state - min(cur_state[0]))/(max(cur_state[0])-min(cur_state[0]) + 1e-10)
            
            cur_state_tensor = torch.FloatTensor(cur_state).float()
            
            # Ensure tensor has batch dimension
            if len(cur_state_tensor.shape) == 1:
                cur_state_tensor = cur_state_tensor.unsqueeze(0)
                
            # print(f"In act() - state tensor shape: {cur_state_tensor.shape}")
                
            # Set actor to evaluation mode
            self.actor.eval()
            with torch.no_grad():
                action = self.actor(cur_state_tensor)
                # If we get a batch dimension, take the first item
                if len(action.shape) > 1:
                    action = action[0]
                action = action.numpy()
            # Set actor back to training mode
            self.actor.train()
            
            print(action)
            action_tmp = action
            action = np.round(action)
            action = action.astype(np.float64)
            flag = 1

        for i in range(action.shape[0]):
            if action[i] <= self.env.default_action[i]:
                print("[Action %d] Lower than DEFAULT: %f" % (i, action[i]))
                action[i] = int(self.env.default_action[i]) * int(self.env.length[i])
            elif action[i] > self.env.a_high[i]:
                print("[Action %d] Higher than MAX: %f" % (i, action[i]))
                action[i] = int(self.env.a_high[i]) * int(self.env.length[i])
            else:
                action[i] = action[i] * self.env.length[i]

        action = self.get_calculate_knobs(action)

        return action, flag, action_tmp