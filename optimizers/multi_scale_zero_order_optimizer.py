import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import time
import json
import torch.nn.functional as F

# Import the registry
from registry import BaseOptimizer, OptimizerRegistry
from torch_model import Actor, Critic
from optimizers.zero_order_optimizer import ZeroOrderOptimizer


@OptimizerRegistry.register("multi_scale_zero_order")
class MultiScaleZeroOrderOptimizer(ZeroOrderOptimizer):
    """Multi-Scale Zero-order optimization algorithm for database parameter tuning.
    
    Implements the multi-scale perturbation mechanism with adaptive weights
    and invalid sampling elimination for improved exploration and convergence.
    """
    
    def __init__(self, env, learning_rate=0.001, train_min_size=32, 
                 size_mem=2000, size_predict_mem=2000, **kwargs):
        """Initialize the multi-scale zero-order optimizer.
        
        Args:
            env: The environment to optimize
            learning_rate: Learning rate for optimization
            train_min_size: Minimum batch size for training
            size_mem: Memory size for experience replay
            size_predict_mem: Memory size for prediction
            **kwargs: Additional optimizer-specific arguments
        """
        # Initialize the base ZeroOrderOptimizer
        super().__init__(env, learning_rate=learning_rate, 
                         train_min_size=train_min_size, 
                         size_mem=size_mem, 
                         size_predict_mem=size_predict_mem,
                         **kwargs)
        
        # Ensure environment reference is set for critics (might be already set by parent class)
        if hasattr(self, 'critic') and not hasattr(self.critic, 'env'):
            self.critic.env = env
        if hasattr(self, 'target_critic') and not hasattr(self.target_critic, 'env'):
            self.target_critic.env = env
        
        # Get configuration values from kwargs or use defaults
        config_dict = kwargs.get("config_dict", {})
        multi_scale_config = config_dict.get("multi_scale_zero_order", {})
        
        try:
            # Multi-scale specific parameters
            self.perturbation_radii = list(map(float, multi_scale_config.get("perturbation_radii", "0.01,0.02,0.05").split(',')))
            self.samples_per_radius = int(multi_scale_config.get("samples_per_radius", 10))
            self.weight_update_interval = int(multi_scale_config.get("weight_update_interval", 20))
            self.history_window_size = int(multi_scale_config.get("history_window_size", 10))
            self.phase_switch_threshold = float(multi_scale_config.get("phase_switch_threshold", 0.6))
            self.variance_threshold = float(multi_scale_config.get("variance_threshold", 0.01))
            self.smooth_strategy = multi_scale_config.get("smooth_strategy", "softmax_smooth")
            self.elimination_offset = float(multi_scale_config.get("elimination_offset", 0.5))
            self.history_retention_rate = float(multi_scale_config.get("history_retention_rate", 0.7))
            self.noise_std = float(multi_scale_config.get("noise_std", 0.15))  # 增大初始扰动范围
            self.noise_decay = float(multi_scale_config.get("noise_decay", 0.995))  # 减缓噪声衰减速度
            self.reward_scale = float(multi_scale_config.get("reward_scale", 1.5))  # 增加奖励缩放因子
        except Exception as e:
            # If any error occurs reading config, use default values
            print(f"Error reading multi_scale_zero_order config, using defaults: {e}")
            self.perturbation_radii = [0.01, 0.02, 0.05]
            self.samples_per_radius = 10
            self.weight_update_interval = 20
            self.history_window_size = 10
            self.phase_switch_threshold = 0.6
            self.variance_threshold = 0.01
            self.smooth_strategy = "softmax_smooth"
            self.elimination_offset = 0.5
            self.history_retention_rate = 0.7
            self.noise_std = 0.15  # 增大初始扰动范围
            self.noise_decay = 0.995  # 减缓噪声衰减速度
            self.reward_scale = 1.5  # 增加奖励缩放因子
        
        # Initialize weights for each perturbation radius (equal initially)
        self.radius_weights = [1.0 / len(self.perturbation_radii)] * len(self.perturbation_radii)
        
        # History of function values for phase detection
        self.func_value_history = deque(maxlen=self.history_window_size)
        
        # Effectiveness scores for each radius
        self.effectiveness_scores = [0.0] * len(self.perturbation_radii)
        
        # Replace the zero_order_opt with multi-scale version
        self.multi_scale_opt = MultiScaleZeroOrderAlgorithm(
            model=self.actor,
            learning_rate=learning_rate,
            perturbation_radii=self.perturbation_radii,
            weight_update_interval=self.weight_update_interval,
            noise_std=self.noise_std,
            noise_decay=self.noise_decay
        )
        
        # Log multi-scale specific configuration
        self.log_json("multi_scale_config", {
            "perturbation_radii": self.perturbation_radii,
            "samples_per_radius": self.samples_per_radius,
            "weight_update_interval": self.weight_update_interval,
            "history_window_size": self.history_window_size,
            "phase_switch_threshold": self.phase_switch_threshold,
            "variance_threshold": self.variance_threshold,
            "smooth_strategy": self.smooth_strategy,
            "elimination_offset": self.elimination_offset,
            "history_retention_rate": self.history_retention_rate,
            "noise_std": self.noise_std,
            "noise_decay": self.noise_decay,
            "reward_scale": self.reward_scale
        })
        
        # 更改训练过程中使用的critic优化器学习率
        if hasattr(self, 'critic_optimizer'):
            for param_group in self.critic_optimizer.param_groups:
                param_group['lr'] = learning_rate * 1.5  # 提高critic学习率

    def _train_actor(self, samples, i):
        """Train the actor network using multi-scale zero-order optimization.
        
        Args:
            samples: Batch of experiences from memory
            i: Current iteration
        """
        # Get the current state from the first sample for evaluation
        for sample in samples:
            cur_state, _, _, _, _ = sample

                        # Normalize state
            if len(cur_state.shape) > 1 and cur_state.shape[0] > 0:
                cur_state = (cur_state - min(cur_state[0])) / (max(cur_state[0]) - min(cur_state[0]) + 1e-10)
            
            # Convert to tensor
            cur_state_tensor = torch.FloatTensor(cur_state).float()


            
            if len(cur_state_tensor.shape) == 1:
                cur_state_tensor = cur_state_tensor.unsqueeze(0)
            
            # Evaluate current policy (without perturbation)
            self.actor.eval()
            with torch.no_grad():
                current_action = self.actor(cur_state_tensor).detach()
            self.actor.train()
            
            # Get current policy value
            self.critic.eval()
            current_value = self.critic(cur_state_tensor, current_action).item()
            self.critic.train()
            
            # Add current function value to history
            self.func_value_history.append(current_value)
            
            # Prepare for multi-scale perturbation
            all_populations = []
            all_rewards = []
            all_variances = []
            
            # Determine the phase for radius selection
            max_steps = getattr(self.env, 'max_steps', 500)
            is_exploration_phase = (
                i <= self.phase_switch_threshold * max_steps and
                (len(self.func_value_history) < self.history_window_size or 
                np.var(list(self.func_value_history)) >= self.variance_threshold)
            )
            
            # Log the current phase
            self.log_training({
                "is_exploration_phase": is_exploration_phase,
                "iteration": i
            }, i)
            
            # For each perturbation radius
            for radius_idx, radius in enumerate(self.perturbation_radii):
                # Generate population for this radius
                population = self.multi_scale_opt.generate_population(
                    npop=self.samples_per_radius, 
                    radius_idx=radius_idx
                )
                all_populations.append(population)
                
                # Evaluate each model in this population
                rewards = []
                for model in population:
                    # Evaluate perturbed model
                    model.eval()
                    with torch.no_grad():
                        perturbed_action = model(cur_state_tensor).detach()
                    model.train()
                    
                    # Get critic value for this action
                    self.critic.eval()
                    critic_value = self.critic(cur_state_tensor, perturbed_action).item()
                    self.critic.train()
                    
                    # Use improvement over current policy as reward with scaling
                    reward = (critic_value - current_value) * self.reward_scale
                    rewards.append(reward)
                
                # Calculate variance for this radius
                variance = np.var(rewards)
                all_rewards.append(rewards)
                all_variances.append(variance)
                
                # Update effectiveness score
                avg_reward = np.mean(rewards)
                self.effectiveness_scores[radius_idx] = (
                    (1 - self.history_retention_rate) * self.effectiveness_scores[radius_idx] + 
                    self.history_retention_rate * avg_reward
                )
            
            # Log evaluation metrics
            self.log_training({
                "radius_rewards": [float(np.mean(r)) for r in all_rewards],
                "radius_variances": [float(v) for v in all_variances],
                "effectiveness_scores": [float(s) for s in self.effectiveness_scores],
                "stage": "population_evaluation"
            }, i)
            
            # Update weights if at update interval
            if i % self.weight_update_interval == 0:
                self._update_radius_weights(i, all_variances)
            
            # Find valid perturbation radii
            valid_radii = self._eliminate_invalid_samples()
            
            # 创建批量更新，避免每个半径单独更新造成的冲突
            all_updates = []
            all_update_weights = []
            
            # 收集所有有效半径的更新方向和对应权重
            for radius_idx in valid_radii:
                rewards = np.array(all_rewards[radius_idx])
                weight = self.radius_weights[radius_idx]
                
                # 归一化奖励
                if len(rewards) > 1:
                    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
                
                # 收集此半径的更新
                all_updates.append((radius_idx, rewards))
                all_update_weights.append(weight)
            
            # 批量应用所有更新 - 使用actor模型的参数
            if all_updates:
                self._apply_combined_updates_to_actor(all_updates, all_update_weights, all_populations)
            
            # Log update information
            self.log_training({
                "valid_radii": valid_radii,
                "radius_weights": [float(w) for w in self.radius_weights],
                "stage": "model_update",
                "batch_update": True
            }, i)
    
    def _apply_combined_updates_to_actor(self, all_updates, all_update_weights, all_populations):
        """应用多个扰动半径的组合更新到actor模型
        
        Args:
            all_updates: 所有更新的列表，每项为(radius_idx, rewards)
            all_update_weights: 对应的权重列表
            all_populations: 所有半径的模型群体列表
        """
        # 遍历actor模型的每个参数
        for i_param, param in enumerate(self.actor.parameters()):
            combined_update = torch.zeros_like(param.data)
            
            # 为每个扰动半径累积更新
            for (radius_idx, rewards), weight in zip(all_updates, all_update_weights):
                perturbation_radius = self.perturbation_radii[radius_idx]
                population = all_populations[radius_idx]
                
                # 获取此半径特定的学习率
                lr = self.multi_scale_opt.get_lr_for_radius(radius_idx)
                
                # 计算此参数在此半径下的更新
                w_update = torch.zeros_like(param.data)
                for j, model in enumerate(population):
                    w_update = w_update + (model.E[i_param] * rewards[j])
                
                # 缩放更新并添加到组合更新中
                scaled_update = (lr * weight / (len(population) * perturbation_radius)) * w_update
                combined_update += scaled_update
            
            # 直接应用组合更新到参数
            param.data.add_(combined_update)
        
        # 更新迭代计数器
        self.multi_scale_opt._count += 1

    def _update_radius_weights(self, iteration, all_variances):
        """Update weights for different perturbation radii based on adaptive strategy.
        
        Args:
            iteration: Current iteration
            all_variances: Variances of rewards for each radius
        """
        # Calculate variance of historical function values
        if len(self.func_value_history) >= self.history_window_size:
            history_variance = np.var(list(self.func_value_history))
        else:
            history_variance = float('inf')  # Default to exploration if not enough data
        
        # Determine the current phase
        # Use max_steps if available, otherwise estimate based on environment
        max_steps = getattr(self.env, 'max_steps', 500)
        is_exploration_phase = (
            iteration <= self.phase_switch_threshold * max_steps and
            history_variance >= self.variance_threshold
        )
        
        new_weights = []
        
        if is_exploration_phase:
            # Exploration phase: favor larger perturbation radii
            if self.smooth_strategy == "linear_smooth":
                # Linear interpolation smoothing
                lambda_val = np.random.uniform(0.05, 0.1)
                for radius in self.perturbation_radii:
                    weight = radius + lambda_val
                    new_weights.append(weight)
            else:  # softmax_smooth
                # Softmax smoothing - 增强大扰动半径的权重
                psi = np.random.uniform(5, 20)  # 增大psi范围，使大扰动半径权重更高
                for radius in self.perturbation_radii:
                    weight = np.exp(psi * radius)
                    new_weights.append(weight)
        else:
            # Convergence phase: balance based on variance and radius
            rho = 100  # Coefficient for radius penalty 
            epsilon = 1e-8  # Small constant to avoid division by zero
            
            for idx, radius in enumerate(self.perturbation_radii):
                variance = max(all_variances[idx], epsilon)
                # 减弱对小半径的惩罚，使其在收敛阶段获得更多权重
                if radius == min(self.perturbation_radii):
                    weight = np.exp(-rho * 0.5 * (radius ** 2)) / (variance + epsilon)
                else:
                    weight = np.exp(-rho * (radius ** 2)) / (variance + epsilon)
                new_weights.append(weight)
        
        # Normalize weights
        total_weight = sum(new_weights)
        self.radius_weights = [w / total_weight for w in new_weights]
        
        # Log weight update
        self.log_json("weight_update", {
            "iteration": iteration,
            "is_exploration_phase": is_exploration_phase,
            "history_variance": float(history_variance),
            "new_weights": [float(w) for w in self.radius_weights]
        })

    def _eliminate_invalid_samples(self):
        """Eliminate invalid samples based on effectiveness scores.
        
        Returns:
            list: Indices of valid perturbation radii
        """
        valid_radii = []
        
        # Calculate mean and standard deviation of effectiveness scores
        mean_score = np.mean(self.effectiveness_scores)
        std_score = np.std(self.effectiveness_scores)
        
        # Calculate threshold 
        threshold = mean_score - self.elimination_offset * std_score
        
        # Find valid perturbation radii
        for idx, score in enumerate(self.effectiveness_scores):
            if score >= threshold:
                valid_radii.append(idx)
        
        # If all samples are eliminated, use the one with highest score
        if not valid_radii and self.effectiveness_scores:
            best_idx = np.argmax(self.effectiveness_scores)
            valid_radii = [best_idx]
        
        # 如果有多个有效半径，总是保留最大的半径(用于探索)和最小的半径(用于精确调优)
        if len(valid_radii) > 1 and len(self.perturbation_radii) > 2:
            min_radius_idx = np.argmin(self.perturbation_radii)
            max_radius_idx = np.argmax(self.perturbation_radii)
            
            if min_radius_idx not in valid_radii:
                valid_radii.append(min_radius_idx)
            if max_radius_idx not in valid_radii:
                valid_radii.append(max_radius_idx)
        
        return valid_radii


class MultiScaleZeroOrderAlgorithm:
    """Multi-scale zero-order optimization algorithm implementation."""
    
    def __init__(self, model, learning_rate, perturbation_radii, weight_update_interval=20, 
                 noise_std=0.15, noise_decay=0.995):
        """Initialize the multi-scale zero-order optimization algorithm.
        
        Args:
            model: The model to optimize
            learning_rate: Learning rate for optimization
            perturbation_radii: List of perturbation radii to use
            weight_update_interval: Interval for updating weights
            noise_std: 扰动参数的初始标准差
            noise_decay: 扰动参数的衰减率
        """
        self.model = model
        self.learning_rate = learning_rate
        self.perturbation_radii = perturbation_radii
        self.weight_update_interval = weight_update_interval
        self._noise_std = noise_std
        self.noise_decay = noise_decay
        self._populations = [None] * len(perturbation_radii)
        self._count = 0
        
        # 为每个半径使用不同的学习率
        self.radius_learning_rates = {}
        for idx, radius in enumerate(perturbation_radii):
            # 小扰动半径使用更小的学习率，大扰动半径使用更大的学习率
            if radius == min(perturbation_radii):
                self.radius_learning_rates[idx] = learning_rate * 0.8
            elif radius == max(perturbation_radii):
                self.radius_learning_rates[idx] = learning_rate * 1.2
            else:
                self.radius_learning_rates[idx] = learning_rate
        
        # Get the environment's action space bounds from the model
        self.env = None
        if hasattr(model, 'env'):
            self.env = model.env
        elif hasattr(model, '_env'):
            self.env = model._env

    @property
    def noise_std(self):
        """Get the current noise standard deviation with decay applied."""
        step_decay = np.power(self.noise_decay, np.floor((1 + self._count) / 50))
        return self._noise_std * step_decay

    def get_population(self, radius_idx):
        """获取指定半径的模型群体
        
        Args:
            radius_idx: 扰动半径索引
            
        Returns:
            list: 模型群体
        """
        return self._populations[radius_idx]
    
    def get_lr_for_radius(self, radius_idx):
        """获取指定半径的学习率
        
        Args:
            radius_idx: 扰动半径索引
            
        Returns:
            float: 学习率
        """
        return self.radius_learning_rates[radius_idx]

    def generate_population(self, npop=10, radius_idx=0):
        """Generate a population of perturbed models for the given radius.
        
        Args:
            npop: Number of models in the population
            radius_idx: Index of the perturbation radius to use
            
        Returns:
            list: A list of perturbed models
        """
        perturbation_radius = self.perturbation_radii[radius_idx]
        population = []
        
        # Store weights of the original model
        original_weights = [param.data.clone() for param in self.model.parameters()]
        
        for _ in range(npop):
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
                # 使用全局噪声方差，而不是半径特定值
                param.data = param.data + self.noise_std * perturbation_radius * noise
            
            population.append(new_model)
        
        # Store the population for this radius
        self._populations[radius_idx] = population
        
        return population

    def update_population(self, rewards, radius_idx=0, weight=1.0):
        """Update the model based on the rewards from the population.
        
        Args:
            rewards: Array of rewards for each model in the population
            radius_idx: Index of the perturbation radius used
            weight: Weight to apply to the update from this radius
        """
        if self._populations[radius_idx] is None:
            raise ValueError(f"Population for radius_idx {radius_idx} is none, generate & eval it first")

        perturbation_radius = self.perturbation_radii[radius_idx]
        population = self._populations[radius_idx]
        
        # Normalize rewards
        if len(rewards) > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # 获取此半径特定的学习率
        lr = self.radius_learning_rates.get(radius_idx, self.learning_rate)

        for i, param in enumerate(self.model.parameters()):
            w_updates = torch.zeros_like(param)
            for j, model in enumerate(population):
                w_updates = w_updates + (model.E[i] * rewards[j])
            
            # Apply the update with the radius-specific weight
            param.data = param.data + (lr * weight / (len(rewards) * perturbation_radius)) * w_updates
        
        self._count = self._count + 1

    def get_model(self):
        """Get the optimized model.
        
        Returns:
            The optimized model
        """
        return self.model