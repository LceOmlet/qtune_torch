import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import time
import json

# Import the registry
from registry import BaseOptimizer, OptimizerRegistry
from torch_model import Actor, Critic


@OptimizerRegistry.register("adam")
class AdamOptimizer(BaseOptimizer):
    """Adam optimization algorithm for database parameter tuning using Actor-Critic method."""
    
    def __init__(self, env, learning_rate=0.001, train_min_size=32, 
                size_mem=2000, size_predict_mem=2000, gamma=0.095, 
                tau=0.125, epsilon=0.9, epsilon_decay=0.999, **kwargs):
        """Initialize the Adam optimizer with Actor-Critic architecture.
        
        Args:
            env: The environment to optimize
            learning_rate: Learning rate for optimization
            train_min_size: Minimum batch size for training
            size_mem: Memory size for experience replay
            size_predict_mem: Memory size for prediction
            gamma: Discount factor for future rewards
            tau: Target network update rate
            epsilon: Exploration rate
            epsilon_decay: Exploration rate decay
            **kwargs: Additional optimizer-specific arguments
        """
        super().__init__(env, **kwargs)
        
        self.learning_rate = learning_rate
        self.train_min_size = train_min_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.tau = tau
        
        # Log the initialization parameters
        self.log_json("init_config", {
            "learning_rate": learning_rate,
            "train_min_size": train_min_size,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "gamma": self.gamma,
            "tau": self.tau
        })
        
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
        
        # Initialize target networks with the same weights
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Set up optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Initialize tracking variables
        self.best_params = None
        self.best_throughput = 0

    def remember(self, cur_state, action, reward, new_state, done):
        """Store experience in memory for replay.
        
        Args:
            cur_state: Current state
            action: Action taken
            reward: Reward received
            new_state: New state
            done: Whether the episode is done
        """
        self.memory.append([cur_state, action, reward, new_state, done])
        
        # Log the experience
        self.log_state({
            "state_shape": str(cur_state.shape),
            "action_shape": str(action.shape),
            "reward": reward[0] if isinstance(reward, np.ndarray) else reward,
            "done": done
        }, len(self.memory))

    def _train_critic(self, samples):
        """Train the critic network.
        
        Args:
            samples: Batch of experiences from memory
        """
        total_critic_loss = 0.0
        
        for idx, sample in enumerate(samples):
            cur_state, action, reward, new_state, done = sample
            
            # Normalize state
            if len(cur_state.shape) > 1 and cur_state.shape[0] > 0:
                cur_state = (cur_state - min(cur_state[0])) / (max(cur_state[0]) - min(cur_state[0]) + 1e-10)
            
            if len(new_state.shape) > 1 and new_state.shape[0] > 0:
                new_state = (new_state - min(new_state[0]))/(max(new_state[0])-min(new_state[0]) + 1e-10)

            # Create tensors
            cur_state_tensor = torch.FloatTensor(cur_state).float()
            action_tensor = torch.FloatTensor(action).float()
            reward_tensor = torch.FloatTensor(reward).float()
            new_state_tensor = torch.FloatTensor(new_state).float()
            done_tensor = torch.FloatTensor(np.array([done]).astype(np.float32))
            
            # Ensure tensors have batch dimension
            if len(cur_state_tensor.shape) == 1:
                cur_state_tensor = cur_state_tensor.unsqueeze(0)
            if len(action_tensor.shape) == 1:
                action_tensor = action_tensor.unsqueeze(0)
            if len(reward_tensor.shape) == 1:
                reward_tensor = reward_tensor.unsqueeze(0)
            if len(new_state_tensor.shape) == 1:
                new_state_tensor = new_state_tensor.unsqueeze(0)
            
            # Get Q-value prediction
            current_q = self.critic(cur_state_tensor, action_tensor)
            
            # Get target Q-value
            with torch.no_grad():
                target_action = self.target_actor(new_state_tensor)
                target_q = self.target_critic(new_state_tensor, target_action)
                target_q = reward_tensor + (1 - done_tensor) * self.gamma * target_q
            
            # Compute critic loss
            critic_loss = nn.MSELoss()(current_q, target_q)
            total_critic_loss += critic_loss.item()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # Log every few samples for debugging
            if idx % 5 == 0:
                self.log_training({
                    "component": "critic",
                    "sample_idx": idx,
                    "critic_loss": float(critic_loss.item()),
                    "current_q": float(current_q.item()),
                    "target_q": float(target_q.item())
                }, len(self.memory))
        
        # Log average critic loss
        avg_critic_loss = total_critic_loss / len(samples)
        self.log_training({
            "component": "critic",
            "avg_critic_loss": float(avg_critic_loss)
        }, len(self.memory))
        
        return avg_critic_loss

    def _train_actor(self, samples):
        """Train the actor network.
        
        Args:
            samples: Batch of experiences from memory
        """
        total_actor_loss = 0.0
        
        for idx, sample in enumerate(samples):
            cur_state, action, reward, new_state, done = sample
            
            # Normalize state
            if len(cur_state.shape) > 1 and cur_state.shape[0] > 0:
                cur_state = (cur_state - min(cur_state[0])) / (max(cur_state[0]) - min(cur_state[0]) + 1e-10)
            
            # Create tensors
            cur_state_tensor = torch.FloatTensor(cur_state).float()
            
            # Ensure tensor has batch dimension
            if len(cur_state_tensor.shape) == 1:
                cur_state_tensor = cur_state_tensor.unsqueeze(0)
            
            # Get predicted action
            predicted_action = self.actor(cur_state_tensor)
            
            # Calculate critic gradients
            critic_value = self.critic(cur_state_tensor, predicted_action)
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss = -torch.mean(critic_value)
            actor_loss.backward()

            # for param in self.actor.parameters():
            #     print("actor grad:", param.grad)

            self.actor_optimizer.step()
            
            total_actor_loss += actor_loss.item()

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("future_reward:", critic_value)
            print("reward:", reward)
            print("cur_state_tensor:", cur_state_tensor)
            print("target_action:", predicted_action.detach().numpy())
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            
            # Log every few samples for debugging
            if idx % 5 == 0:
                self.log_training({
                    "component": "actor",
                    "sample_idx": idx,
                    "actor_loss": float(actor_loss.item()),
                    "critic_value": float(critic_value.item())
                }, len(self.memory))
        
        # Log average actor loss
        avg_actor_loss = total_actor_loss / len(samples)
        self.log_training({
            "component": "actor",
            "avg_actor_loss": float(avg_actor_loss)
        }, len(self.memory))
        
        return avg_actor_loss

    def _update_target_networks(self):
        """Update target networks with a soft update."""
        # Update target actor
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Update target critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # Log update
        self.log_training({
            "component": "target_networks",
            "update": "soft_update",
            "tau": self.tau
        }, len(self.memory))

    def train(self, i=0):
        """Train the actor and critic networks.
        
        Args:
            i: Current iteration (used for epsilon decay)
        """
        batch_size = min(len(self.memory), 32)
        
        if len(self.memory) < self.train_min_size:
            return
        
        # Log training start
        self.log_training({
            "stage": "training_start",
            "batch_size": batch_size,
            "memory_size": len(self.memory),
            "epsilon": self.epsilon
        }, i)
        
        # Sample random batch from memory
        indexes = np.random.choice(len(self.memory), batch_size, replace=False)
        samples = [self.memory[i] for i in indexes]
        
        # Train critic
        critic_loss = self._train_critic(samples)
        
        # Train actor
        actor_loss = self._train_actor(samples)
        
        # Update target networks
        self._update_target_networks()
        
        # Decay epsilon for exploration
        if i > 0:
            old_epsilon = self.epsilon
            self.epsilon *= self.epsilon_decay
            
            # Log epsilon decay
            self.log_training({
                "component": "exploration",
                "old_epsilon": old_epsilon,
                "new_epsilon": self.epsilon,
                "decay_factor": self.epsilon_decay
            }, i)
        
        # Log training completion
        self.log_training({
            "stage": "training_complete",
            "critic_loss": critic_loss,
            "actor_loss": actor_loss
        }, i)

    def act(self, state):
        """Choose an action based on the current state using epsilon-greedy strategy.
        
        Args:
            state: Current state
            
        Returns:
            tuple: (action, is_predicted, action_tmp)
        """
        # Normalize state
        if len(state.shape) > 1 and state.shape[0] > 0:
            state = (state - min(state[0])) / (max(state[0]) - min(state[0]) + 1e-10)
        
        # Create tensor
        state_tensor = torch.FloatTensor(state).float()
        
        # Ensure tensor has batch dimension
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            # Random action (exploration)
            is_predicted = 0
            action = np.random.uniform(
                self.env.a_low, 
                self.env.a_high, 
                size=self.env.action_space.shape[0]
            )
            action_tmp = np.zeros_like(action)
            
            # Log random action
            self.log_action({
                "action_type": "random",
                "epsilon": self.epsilon,
                "action_shape": str(action.shape)
            }, len(self.memory))
        else:
            # Use actor to predict action (exploitation)
            is_predicted = 1
            self.actor.eval()
            with torch.no_grad():
                action = self.actor(state_tensor).detach().numpy().flatten()
            self.actor.train()
            action_tmp = action.copy()
            
            # Log model action
            self.log_action({
                "action_type": "model",
                "epsilon": self.epsilon,
                "action_shape": str(action.shape)
            }, len(self.memory))
        
        return action, is_predicted, action_tmp

    def optimize(self, num_trials: int) -> Tuple[Dict[str, float], float]:
        """Run the Adam-based Actor-Critic optimization algorithm.
        
        Args:
            num_trials: Number of optimization iterations to run
            
        Returns:
            tuple: The best parameters found and the best performance metric achieved
        """
        print("\n------ Starting Database Parameter Tuning with Adam-based Actor-Critic ------\n")
        
        # Log optimization start
        self.log_json("optimization_start", {
            "num_trials": num_trials,
            "state_dim": self.env.observation_space.shape[0],
            "action_dim": self.env.action_space.shape[0]
        })
        
        # Initialize state
        cur_state = self.env._get_obs()
        cur_state = cur_state.reshape((1, self.env.state.shape[0]))
        
        # Initialize action
        action = self.env.fetch_action()
        action_2 = action.reshape((1, self.env.knob_num))
        action_2 = action_2[:, :self.env.action_space.shape[0]]
        
        # Apply first action and get initial state
        new_state, reward, score, cur_throughput = self.env.step(action, 0, 1)
        new_state = new_state.reshape((1, self.env.state.shape[0]))
        reward_np = np.array([reward])
        
        # Store initial experience
        self.remember(cur_state, action_2, reward_np, new_state, 0)
        
        # Log initial action and result
        self.log_evaluation({
            "iteration": 0,
            "action_type": "initial",
            "reward": reward,
            "score": score,
            "throughput": cur_throughput
        }, 0)
        
        # Initialize best performance tracking
        self.best_throughput = cur_throughput
        self.best_params = {k: v for k, v in zip(self.env.db.knob_names, action)}
        
        # Log initial best performance
        self.log_json("initial_best", {
            "throughput": self.best_throughput,
            "params": self.best_params
        })
        
        # Main optimization loop
        for i in range(1, num_trials + 1):
            # Create new action
            cur_state = new_state
            action, is_predicted, action_tmp = self.act(cur_state)
            
            # Apply action and get new state
            new_state, reward, score, throughput = self.env.step(action, is_predicted, i + 1)
            new_state = new_state.reshape((1, self.env.state.shape[0]))
            
            # Log evaluation results
            self.log_evaluation({
                "is_predicted": is_predicted,
                "reward": reward,
                "score": score,
                "throughput": throughput,
                "action_type": "model" if is_predicted else "random"
            }, i)
            
            # Store experience
            reward_np = np.array([reward])
            action = action.reshape((1, self.env.knob_num))
            action_2 = action[:, :self.env.action_space.shape[0]]
            self.remember(cur_state, action_2, reward_np, new_state, 0)
            
            # Train the model
            self.train(i)
            
            # Update best performance if improved
            if throughput > self.best_throughput:
                improvement = (throughput - self.best_throughput) / self.best_throughput * 100
                self.best_throughput = throughput
                self.best_params = {k: v for k, v in zip(self.env.db.knob_names, action.flatten())}
                
                # Log new best performance
                self.log_json("new_best", {
                    "iteration": i,
                    "throughput": self.best_throughput,
                    "improvement_percentage": improvement,
                    "params": self.best_params
                })
                
                print(f"\nNew best throughput at iteration {i}: {self.best_throughput}")
                print(f"Improvement: {improvement:.2f}%")
            
            # Print progress
            if i % 5 == 0:
                print(f"Iteration {i}/{num_trials} completed. Current throughput: {throughput}, Best so far: {self.best_throughput}")
        
        # Save best parameters to file
        result_file = f'training-results/adam_best_params_{self.timestamp}.json'
        with open(result_file, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        print(f"\n------ Optimization Complete ------")
        print(f"Best throughput: {self.best_throughput}")
        print(f"Best parameters saved to: {result_file}")
        
        # Log final results
        self.log_json("optimization_complete", {
            "num_iterations": i,
            "best_throughput": self.best_throughput,
            "best_params": self.best_params,
            "result_file": result_file
        })
        
        return self.best_params, self.best_throughput 