import numpy as np
import time
import json
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm

# Import the registry
from registry import BaseOptimizer, OptimizerRegistry


# Define acquisition functions
class ExpectedImprovement:
    """Expected Improvement acquisition function."""
    
    def __init__(self, xi=0.01):
        """Initialize the ExpectedImprovement acquisition function.
        
        Args:
            xi: Exploitation-exploration trade-off parameter
        """
        self.xi = xi
    
    def __call__(self, X, gp, y_best):
        """Calculate the expected improvement at points X.
        
        Args:
            X: Points at which to calculate the EI
            gp: Gaussian process model
            y_best: Best observed value
            
        Returns:
            Expected improvement at points X
        """
        mu, sigma = gp.predict(X, return_std=True)
        
        # Expected improvement
        with np.errstate(divide='warn'):
            imp = mu - y_best - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei


class UpperConfidenceBound:
    """Upper Confidence Bound acquisition function."""
    
    def __init__(self, kappa=2.576):
        """Initialize the UpperConfidenceBound acquisition function.
        
        Args:
            kappa: Exploitation-exploration trade-off parameter
        """
        self.kappa = kappa
    
    def __call__(self, X, gp, y_best=None):
        """Calculate the upper confidence bound at points X.
        
        Args:
            X: Points at which to calculate the UCB
            gp: Gaussian process model
            y_best: Not used, kept for API consistency
            
        Returns:
            Upper confidence bound at points X
        """
        mu, sigma = gp.predict(X, return_std=True)
        return mu + self.kappa * sigma


@OptimizerRegistry.register("bayesian")
class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimization algorithm for database parameter tuning."""
    
    def __init__(self, env, learning_rate=None, train_min_size=None, 
                size_mem=None, size_predict_mem=None, config=None, **kwargs):
        """Initialize the Bayesian optimizer.
        
        Args:
            env: The environment to optimize
            learning_rate: Not used by Bayesian optimization, kept for API consistency
            train_min_size: Not used by Bayesian optimization, kept for API consistency
            size_mem: Not used by Bayesian optimization, kept for API consistency
            size_predict_mem: Not used by Bayesian optimization, kept for API consistency
            config: Configuration dictionary for Bayesian optimization
            **kwargs: Additional optimizer-specific arguments
        """
        super().__init__(env, **kwargs)
        
        # Get configuration
        self.config = config if config is not None else {}
        if kwargs.get("config_dict") is not None:
            bayesian_config = kwargs.get("config_dict").get("bayesian", {})
            for key, value in bayesian_config.items():
                if key not in self.config:
                    self.config[key] = value
        
        # Set Bayesian optimization parameters
        self.n_init_points = int(self.config.get("n_init_points", 5))
        self.acq_func_name = self.config.get("acq_func", "ei").lower()
        self.noise_level = float(self.config.get("noise_level", 0.1))
        self.normalize_y = self.config.get("normalize_y", "true").lower() == "true"
        self.kappa = float(self.config.get("kappa", 2.576))
        self.xi = float(self.config.get("xi", 0.01))
        self.population_size = int(self.config.get("population_size", 10))
        self.random_seed = int(self.config.get("random_seed", 42))
        
        # Log configuration
        self.log_json("bayesian_config", {
            "n_init_points": self.n_init_points,
            "acq_func": self.acq_func_name,
            "noise_level": self.noise_level,
            "normalize_y": self.normalize_y,
            "kappa": self.kappa,
            "xi": self.xi,
            "population_size": self.population_size,
            "random_seed": self.random_seed
        })
        
        # Print configuration
        print("\n====== Bayesian Optimizer Configuration ======")
        print(f"  n_init_points: {self.n_init_points}")
        print(f"  acq_func: {self.acq_func_name}")
        print(f"  noise_level: {self.noise_level}")
        print(f"  normalize_y: {self.normalize_y}")
        print(f"  kappa: {self.kappa}")
        print(f"  xi: {self.xi}")
        print(f"  random_seed: {self.random_seed}")
        print("============================================\n")
        
        # Set random number generator
        self.rng = np.random.RandomState(self.random_seed)
        
        # Set up the acquisition function
        if self.acq_func_name == 'ei':
            self.acq_func = ExpectedImprovement(xi=self.xi)
        elif self.acq_func_name == 'ucb':
            self.acq_func = UpperConfidenceBound(kappa=self.kappa)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acq_func_name}")
        
        # Get knob information from environment
        self.knob_names = list(env.knob_config.keys())
        self.parameter_types = []
        self.parameter_bounds = []
        
        for name in self.knob_names:
            knob_info = env.knob_config[name]
            min_val = knob_info['min_value']
            max_val = knob_info['max_value']
            
            # Determine parameter type
            if isinstance(min_val, int) and isinstance(max_val, int):
                self.parameter_types.append('int')
            else:
                self.parameter_types.append('float')
            
            self.parameter_bounds.append((min_val, max_val))
        
        # Log parameter bounds
        self.log_json("parameter_bounds", {
            "names": self.knob_names,
            "types": self.parameter_types,
            "bounds": self.parameter_bounds
        })
        
        # Initialize Gaussian Process
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(len(self.knob_names)), nu=2.5) + WhiteKernel(noise_level=self.noise_level)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,  # Avoid numerical issues
            normalize_y=self.normalize_y,
            n_restarts_optimizer=5,
            random_state=self.random_seed
        )
        
        # Initialize data structures
        self.X = []  # Normalized parameters
        self.X_orig = []  # Original parameters
        self.y = []  # Observed values
        
        # Initialize best parameters
        self.best_params = None
        self.best_value = None
    
    def _normalize_parameters(self, params):
        """Normalize parameters to [0, 1] range.
        
        Args:
            params: Parameters to normalize
            
        Returns:
            Normalized parameters
        """
        norm_params = []
        for i, param in enumerate(params):
            min_val, max_val = self.parameter_bounds[i]
            norm_param = (param - min_val) / (max_val - min_val)
            norm_params.append(norm_param)
        return np.array(norm_params)
    
    def _denormalize_parameters(self, norm_params):
        """Convert normalized parameters back to their original range.
        
        Args:
            norm_params: Normalized parameters
            
        Returns:
            Denormalized parameters
        """
        params = []
        for i, norm_param in enumerate(norm_params):
            min_val, max_val = self.parameter_bounds[i]
            param = norm_param * (max_val - min_val) + min_val
            
            # Round to integer if the parameter type is int
            if self.parameter_types[i] == 'int':
                param = int(round(param))
                
            params.append(param)
        return np.array(params)
    
    def suggest_parameters(self):
        """Suggest parameters to evaluate next.
        
        Returns:
            dict: Dictionary of parameter names to values
        """
        # If we haven't evaluated enough points, choose a random point
        if len(self.X) < self.n_init_points:
            # Generate a random point in the normalized space [0, 1]^d
            x = self.rng.random(len(self.knob_names))
            params = self._denormalize_parameters(x)
            
            # Log random point selection
            self.log_action({
                "selection_method": "random_initial",
                "iteration": len(self.X),
                "normalized_params": x.tolist()
            }, len(self.X))
        else:
            # Use Bayesian optimization to select the next point
            # Fit the GP with the observed data
            self.gp.fit(np.array(self.X), np.array(self.y))
            
            # Log GP hyperparameters
            self.log_training({
                "component": "gaussian_process",
                "iteration": len(self.X),
                "kernel_params": str(self.gp.kernel_.get_params()),
                "log_marginal_likelihood": float(self.gp.log_marginal_likelihood_value_)
            }, len(self.X))
            
            # Find the point that maximizes the acquisition function
            x_best = self._find_max_acquisition()
            params = self._denormalize_parameters(x_best)
            
            # Log Bayesian optimization selection
            self.log_action({
                "selection_method": "bayesian_optimization",
                "iteration": len(self.X),
                "normalized_params": x_best.tolist(),
                "acquisition_function": self.acq_func_name
            }, len(self.X))
        
        # Convert to dictionary
        param_dict = {name: params[i] for i, name in enumerate(self.knob_names)}
        return param_dict
    
    def _find_max_acquisition(self):
        """Find the point that maximizes the acquisition function.
        
        Returns:
            numpy.ndarray: The point that maximizes the acquisition function
        """
        # Define negative acquisition function (for minimization)
        def negative_acq(x):
            # Reshape x for single sample prediction
            x = x.reshape(1, -1)
            
            # Calculate acquisition function value
            if len(self.y) > 0:
                y_best = np.max(self.y)
                acq_val = self.acq_func(x, self.gp, y_best)
            else:
                acq_val = self.acq_func(x, self.gp, 0)
            
            # Return negative value for minimization
            return -acq_val[0]
        
        # Start optimization from random points
        best_x = None
        best_acq_val = np.inf
        
        # Run local optimization from multiple starting points
        n_restarts = self.population_size
        for _ in range(n_restarts):
            # Generate a random starting point
            x0 = self.rng.random(len(self.knob_names))
            
            # Run local optimization
            result = minimize(
                negative_acq,
                x0,
                bounds=[(0, 1)] * len(self.knob_names),
                method="L-BFGS-B"
            )
            
            # Update best point if better
            if result.fun < best_acq_val:
                best_acq_val = result.fun
                best_x = result.x
                
        # Log optimization result
        self.log_training({
            "component": "acquisition_optimization",
            "best_acq_val": float(-best_acq_val),
            "best_normalized_params": best_x.tolist()
        }, len(self.X))
        
        return best_x
    
    def update(self, params, value):
        """Update the Bayesian optimization model with new observation.
        
        Args:
            params: Parameters that were evaluated
            value: Value (e.g., throughput) that was observed
        """
        # Convert params to array if it's a dictionary
        if isinstance(params, dict):
            param_array = np.array([params[name] for name in self.knob_names])
        else:
            param_array = np.array(params)
        
        # Normalize parameters
        norm_params = self._normalize_parameters(param_array)
        
        # Store observation
        self.X.append(norm_params)
        self.X_orig.append(param_array)
        self.y.append(value)
        
        # Update best value
        if self.best_value is None or value > self.best_value:
            self.best_value = value
            self.best_params = {name: param_array[i] for i, name in enumerate(self.knob_names)}
            
            # Log new best value
            self.log_json("new_best", {
                "iteration": len(self.X) - 1,
                "params": self.best_params,
                "value": float(self.best_value)
            })
            
        # Log update
        self.log_evaluation({
            "iteration": len(self.X) - 1,
            "params": param_array.tolist(),
            "normalized_params": norm_params.tolist(),
            "value": float(value),
            "best_value": float(self.best_value)
        }, len(self.X) - 1)
    
    def run_optimization(self, num_trials: int) -> Tuple[Dict[str, float], float]:
        """Run the optimization process.
        
        Args:
            num_trials: Number of optimization iterations to run
            
        Returns:
            tuple: The best parameters found and the best performance metric achieved
        """
        # Log optimization start
        self.log_json("optimization_start", {
            "num_trials": num_trials
        })
        
        print("\n------ Starting Database Parameter Tuning with Bayesian Optimization ------\n")
        
        # Initialize state
        cur_state = self.env._get_obs()
        
        # Get initial default action from environment
        action = self.env.fetch_action()
        
        # Apply first action and get initial state
        new_state, reward, score, throughput = self.env.step(action, 0, 1)
        
        # Create parameter dictionary from default action
        default_params = {name: action[i] for i, name in enumerate(self.knob_names)}
        
        # Log initial state
        self.log_evaluation({
            "iteration": 0,
            "action_type": "default",
            "params": action.tolist(),
            "reward": float(reward),
            "score": float(score),
            "throughput": float(throughput)
        }, 0)
        
        # Update the model with default parameters
        self.update(default_params, throughput)
        
        # Main optimization loop
        for i in range(1, num_trials + 1):
            # Get next parameters to evaluate
            params_dict = self.suggest_parameters()
            
            # Convert to array for environment
            param_array = np.array([params_dict[name] for name in self.knob_names])
            
            # Apply parameters and get results
            new_state, reward, score, throughput = self.env.step(param_array, 1, i + 1)
            
            # Log results
            self.log_evaluation({
                "iteration": i,
                "action_type": "bayesian",
                "params": param_array.tolist(),
                "reward": float(reward),
                "score": float(score),
                "throughput": float(throughput)
            }, i)
            
            # Update the model with new observation
            self.update(params_dict, throughput)
            
            # Print progress
            if i % 5 == 0 or i == 1:
                print(f"Iteration {i}/{num_trials} - Throughput: {throughput:.2f}, Best so far: {self.best_value:.2f}")
        
        # Save best parameters to file
        result_file = f'training-results/bayesian_best_params_{self.timestamp}.json'
        with open(result_file, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        # Log final results
        self.log_json("optimization_complete", {
            "num_iterations": i,
            "best_value": float(self.best_value),
            "best_params": self.best_params,
            "result_file": result_file,
            "all_observations": {
                "params": [x.tolist() for x in self.X_orig],
                "values": [float(y) for y in self.y]
            }
        })
        
        print(f"\n------ Bayesian Optimization Complete ------")
        print(f"Best throughput: {self.best_value:.2f}")
        print(f"Best parameters saved to: {result_file}")
        
        return self.best_params, self.best_value
    
    def optimize(self, num_trials: int) -> Tuple[Dict[str, float], float]:
        """Interface method for optimization (calls run_optimization).
        
        Args:
            num_trials: Number of optimization iterations to run
            
        Returns:
            tuple: The best parameters found and the best performance metric achieved
        """
        return self.run_optimization(num_trials) 