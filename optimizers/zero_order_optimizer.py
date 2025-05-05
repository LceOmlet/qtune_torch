import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import time
import json
import torch.nn.functional as F
import numpy as np
# Import the registry
from registry import BaseOptimizer, OptimizerRegistry
from torch_model import Actor, Critic


@OptimizerRegistry.register("zero_order")
class ZeroOrderOptimizer(BaseOptimizer):
    """Zero-order optimization algorithm for database parameter tuning."""
    
    def __init__(self, env, learning_rate=1e-3, noise_std=None, noise_decay=None, 
                 lr_decay=None, decay_step=None, norm_rewards=None, train_min_size=32, 
                 size_mem=2000, size_predict_mem=2000, **kwargs):
        """Initialize the zero-order optimizer.
        
        Args:
            env: The environment to optimize
            learning_rate: Learning rate for optimization
            noise_std: Standard deviation of noise for perturbation
            noise_decay: Decay rate for noise standard deviation
            lr_decay: Decay rate for learning rate
            decay_step: Number of steps after which to decay learning rate and noise
            norm_rewards: Whether to normalize rewards
            train_min_size: Minimum batch size for training
            size_mem: Memory size for experience replay
            size_predict_mem: Memory size for prediction
            **kwargs: Additional optimizer-specific arguments
        """
        super().__init__(env, **kwargs)
        
        self.learning_rate = learning_rate
        self.train_min_size = train_min_size
        self.epsilon = 0.9  # For exploration
        self.epsilon_decay = 0.999
        self.gamma = 0.095
        self.tau = 0.125
        
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
        
        # Create actor and critic networks (same as in ActorCritic)
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

        # Set environment reference for action normalization in critic networks
        self.critic.env = env
        self.target_critic.env = env

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Get configuration values from kwargs or use defaults
        config_dict = kwargs.get("config_dict", {})
        zero_order_config = config_dict.get("zero_order", {})
        
        try:
            self._noise_std = noise_std if noise_std is not None else float(zero_order_config.get("noise_std", 1e-3))
            self.noise_decay = noise_decay if noise_decay is not None else float(zero_order_config.get("noise_decay", 0.99))
            self.lr_decay = lr_decay if lr_decay is not None else float(zero_order_config.get("lr_decay", 0.99))
            self.decay_step = decay_step if decay_step is not None else int(zero_order_config.get("decay_step", 50))
            self.norm_rewards = norm_rewards if norm_rewards is not None else zero_order_config.get("norm_rewards", "true").lower() == "true"
        except Exception as e:
            # If any error occurs reading from config, use default values
            print(f"Error reading zero_order config, using defaults: {e}")
            self._noise_std = 1e-3 if noise_std is None else noise_std
            self.noise_decay = 0.99 if noise_decay is None else noise_decay
            self.lr_decay = 0.99 if lr_decay is None else lr_decay
            self.decay_step = 50 if decay_step is None else decay_step
            self.norm_rewards = True if norm_rewards is None else norm_rewards
        
        # Log the zero-order specific configuration
        self.log_json("zero_order_config", {
            "noise_std": self._noise_std,
            "noise_decay": self.noise_decay,
            "lr_decay": self.lr_decay,
            "decay_step": self.decay_step,
            "norm_rewards": self.norm_rewards
        })
        
        # Set the environment attribute on the model for the optimizer to access
        self.actor.env = self.env
        
        # Initialize the zero-order optimization algorithm
        self.zero_order_opt = ZeroOrderAlgorithm(
            model=self.actor,
            learning_rate=learning_rate,
            noise_std=self._noise_std,
            noise_decay=self.noise_decay,
            lr_decay=self.lr_decay,
            decay_step=self.decay_step,
            norm_rewards=self.norm_rewards
        )
        
        # Initialize tracking variables
        self.best_params = None
        self.best_throughput = 0

    def remember(self, cur_state, action, reward, new_state, done):
        """Store experience in memory for replay."""
        self.memory.append([cur_state, action, reward, new_state, done])
        
        # Log the experience
        self.log_state({
            "state_shape": str(cur_state.shape),
            "action_shape": str(action.shape),
            "reward": reward[0] if isinstance(reward, np.ndarray) else reward,
            "done": done
        }, len(self.memory))

    def train(self, i=0):
        """Train the model using zero-order optimization.
        
        Args:
            i: Current iteration (used for epsilon decay)
        """
        if len(self.memory) < self.train_min_size:
            return
        
        # Sample batch from memory
        batch_size = min(len(self.memory), 32)
        indexes = np.random.choice(len(self.memory), batch_size, replace=False)
        samples = [self.memory[i] for i in indexes]
        
        # Log training start
        self.log_training({
            "batch_size": batch_size,
            "memory_size": len(self.memory),
            "epsilon": self.epsilon,
            "noise_std": self.zero_order_opt.noise_std,
            "learning_rate": self.zero_order_opt.lr
        }, i)
        
        # Train actor using zero-order optimization
        self._train_critic(samples, i)
        self._train_actor(samples, i)
        self.update_target()

        # Decay epsilon
        if i > 0:
            self.epsilon = self.epsilon * self.epsilon_decay
            
        # Log training end
        self.log_training({
            "epsilon_after": self.epsilon,
            "training_completed": True
        }, i)

        

    def update_target(self):
        # Soft update of target networks
        # print("target_critic before:", list(self.target_critic.parameters())[1].data)
        # print("critic before:", list(self.critic.parameters())[1].data)
        
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # print("target_critic after:", list(self.target_critic.parameters())[1].data)
        # print("critic after:", list(self.critic.parameters())[1].data)

    def _train_critic(self, samples, i):
        """Train the critic network.
        
        Args:
            samples: Batch of experiences from memory
        """
        total_critic_loss = 0.0
        
        # Print critic parameters before training
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
            self.target_critic.eval()
            with torch.no_grad():
                target_action = self.target_actor(new_state_tensor).detach()
                future_reward = self.target_critic(new_state_tensor, target_action)[0][0].detach().numpy()
            self.target_critic.train()

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("future_reward:", future_reward)
            print("reward:", reward)
            reward += self.gamma * future_reward
            print("reward:", reward)
            
            # 修改target_value的构建方式，确保精度和类型正确
            target_value = torch.tensor([[float(reward)]], dtype=torch.float32, requires_grad=False)
            
            # 确保critic处于训练模式
            self.critic.train()
            
            # Ensure tensors have requires_grad=True for gradient computation
            cur_state_tensor.requires_grad = True
            action_tensor.requires_grad = True
            
            critic_value = self.critic(cur_state_tensor, action_tensor)
            
            critic_loss = nn.MSELoss()(critic_value, target_value)
                
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)  # Add retain_graph=True to keep gradients
            
            
            # 梯度裁剪，防止梯度消失或爆炸
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            
            self.critic_optimizer.step()
            print("critic_value:", critic_value)
            print("loss:", critic_loss)

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            
            total_critic_loss += critic_loss.item()

            # Update critic
            # Log every few samples for debugging
            if idx % 5 == 0:
                self.log_training({
                    "component": "critic",
                    "sample_idx": idx,
                    "critic_loss": float(critic_loss.item()),
                    "future_reward": float(future_reward),
                    "target_value": float(target_value.numpy())
                }, len(self.memory))
                    
        # Log average critic loss
        avg_critic_loss = total_critic_loss / len(samples)
        self.log_training({
            "component": "critic",
            "avg_critic_loss": float(avg_critic_loss)
        }, len(self.memory))
        
        return avg_critic_loss

    def _train_actor(self, samples, i):
        """Train the actor network using zero-order optimization.
        
        Args:
            samples: Batch of experiences from memory
            i: Current iteration
        """
        total_actor_loss = 0.0

        for idx, sample in enumerate(samples):
            cur_state, action, reward, new_state, _ = sample

            # Normalize state
            if len(cur_state.shape) > 1 and cur_state.shape[0] > 0:
                cur_state = (cur_state - min(cur_state[0]))/(max(cur_state[0])-min(cur_state[0]) + 1e-10)

            cur_state_tensor = torch.FloatTensor(cur_state).float()
            action_tensor = self.actor(cur_state_tensor)
            reward_original = self.critic(cur_state_tensor, action_tensor)
            # Generate population for zero-order optimization
            population = self.zero_order_opt.generate_population(npop=len(samples))
            rewards = []
            
            # Log population generation
            self.log_training({
                "population_size": len(population),
                "stage": "population_generation"
            }, i)
            
            # Evaluate each model in population (following original implementation)
            for model_idx, model in enumerate(population):
                # Use first sample for evaluation
                
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
                
                # Set critic back to training mode
                self.critic.train()
                
                # Use mean of gradients as reward

                rewards.append((critic_value - reward_original).detach().numpy().item())
                
                # Log model evaluation
                if model_idx % 10 == 0:  # Log every 10th model to avoid excessive logging
                    self.log_training({
                        "model_idx": model_idx,
                        "critic_value": float(critic_value.item()),
                        "stage": "model_evaluation"
                    }, i)
            
            # Log rewards statistics before update
            self.log_training({
                "rewards_mean": float(np.mean(rewards)),
                "rewards_std": float(np.std(rewards)),
                "rewards_min": float(np.min(rewards)),
                "rewards_max": float(np.max(rewards)),
                "stage": "before_update"
            }, i)
            
            # Update population using zero-order optimization
            self.zero_order_opt.update_population(np.array(rewards))
            self.actor = self.zero_order_opt.get_model()
            
            # Log after update
            self.log_training({
                "stage": "after_update",
                "update_complete": True
            }, i)

                # Log average actor loss
        avg_actor_loss = total_actor_loss / len(samples)
        self.log_training({
            "component": "actor",
            "avg_actor_loss": float(avg_actor_loss)
        }, len(self.memory))
        
        return avg_actor_loss

    def act(self, state):
        """Choose an action based on the current state using epsilon-greedy strategy.
        
        Args:
            state: Current state
            
        Returns:
            tuple: (action, is_predicted, action_tmp)
        """
        # Normalize state
        if len(state.shape) > 1 and state.shape[0] > 0:
            state = (state - min(state[0]))/(max(state[0])-min(state[0]) + 1e-10)
        
        state_tensor = torch.FloatTensor(state).float()
        
        # Ensure tensor has batch dimension
        if len(state_tensor.shape) == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        if np.random.random() < self.epsilon:
            # Random action
            is_predicted = 0
            action = np.random.uniform(
                self.env.a_low, 
                self.env.a_high, 
                size=self.env.action_space.shape[0]
            )
            action_tmp = np.zeros_like(action)
            
            # Log random action selection
            self.log_action({
                "action_type": "random",
                "epsilon": self.epsilon,
                "action_shape": str(action.shape)
            }, len(self.memory))
        else:
            # Use actor to predict action
            is_predicted = 1
            self.actor.eval()
            with torch.no_grad():
                action = self.actor(state_tensor).detach().numpy().flatten()
            self.actor.train()
            action_tmp = action.copy()
            
            # Log model-based action selection
            self.log_action({
                "action_type": "model",
                "epsilon": self.epsilon,
                "action_shape": str(action.shape)
            }, len(self.memory))
        
        return action, is_predicted, action_tmp

    def optimize(self, num_trials: int) -> Tuple[Dict[str, float], float]:
        """Run the zero-order optimization algorithm.
        
        Args:
            num_trials: Number of optimization iterations to run
            
        Returns:
            tuple: The best parameters found and the best performance metric achieved
        """
        print("\n------ Starting Database Parameter Tuning with Zero-Order Optimization ------\n")
        
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
        result_file = f'training-results/zero_order_best_params_{self.timestamp}.json'
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


class ZeroOrderAlgorithm:
    """Zero-order optimization algorithm implementation."""
    
    def __init__(self, model, learning_rate, noise_std=0.1, noise_decay=0.99, 
                 lr_decay=0.99, decay_step=50, norm_rewards=True):
        """Initialize the zero-order optimization algorithm.
        
        Args:
            model: The model to optimize
            learning_rate: Learning rate for optimization
            noise_std: Standard deviation of noise for perturbation
            noise_decay: Decay rate for noise standard deviation
            lr_decay: Decay rate for learning rate
            decay_step: Number of steps after which to decay learning rate and noise
            norm_rewards: Whether to normalize rewards
        """
        self.model = model
        self._lr = learning_rate
        self._noise_std = noise_std
        self.noise_decay = noise_decay
        self.lr_decay = lr_decay
        self.decay_step = decay_step
        self.norm_rewards = norm_rewards
        self._population = None
        self._count = 0
        
        # Get the environment's action space bounds from the model
        self.env = None
        if hasattr(model, 'env'):
            self.env = model.env
        elif hasattr(model, '_env'):
            self.env = model._env

    @property
    def noise_std(self):
        """Get the current noise standard deviation with decay applied."""
        step_decay = np.power(self.noise_decay, np.floor((1 + self._count) / self.decay_step))
        return self._noise_std * step_decay

    @property
    def lr(self):
        """Get the current learning rate with decay applied."""
        step_decay = np.power(self.lr_decay, np.floor((1 + self._count) / self.decay_step))
        return self._lr * step_decay

    def generate_population(self, npop=50):
        """Generate a population of perturbed models for zero-order optimization.
        
        Args:
            npop: Number of models in the population
            
        Returns:
            list: A list of perturbed models
        """
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
        """Update the model based on the rewards from the population.
        
        Args:
            rewards: Array of rewards for each model in the population
        """
        if self._population is None:
            raise ValueError("populations is none, generate & eval it first")

        if self.norm_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        for i, param in enumerate(self.model.parameters()):
            w_updates = torch.zeros_like(param)
            for j, model in enumerate(self._population):
                w_updates = w_updates + (model.E[i] * rewards[j])
            updates = (self.lr / (len(rewards) * self.noise_std)) * w_updates
            param.data = param.data + updates
            # print("param.data:", param.data)
            # print("updates:", updates)
        
        self._count = self._count + 1

    def get_model(self):
        """Get the optimized model.
        
        Returns:
            The optimized model
        """
        return self.model 