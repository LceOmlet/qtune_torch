import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import json
from configs import knob_config, config_dict
from scipy import special

class ExpectedImprovement:
    """Expected Improvement acquisition function"""
    def __init__(self, xi=0.01):
        self.xi = xi
        
    def __call__(self, X, model, y_best):
        """
        Args:
            X: Points to evaluate EI at
            model: GP model
            y_best: Current best observed value
        """
        mu, sigma = model.predict(X, return_std=True)
        
        # 保持sigma为一维数组，与mu形状相同
        # sigma = sigma.reshape(-1, 1)  # 这行导致了维度不匹配问题
        
        # Expected improvement
        with np.errstate(divide='warn'):
            imp = mu - y_best - self.xi
            Z = np.divide(imp, sigma, out=np.zeros_like(sigma), where=sigma!=0)
            ei = imp * self.norm_cdf(Z) + sigma * self.norm_pdf(Z)
            ei[sigma < 1e-6] = 0.0
            
        return ei
    
    @staticmethod    
    def norm_cdf(x):
        return 0.5 * (1 + special.erf(x / np.sqrt(2)))
    
    @staticmethod
    def norm_pdf(x):
        return np.exp(-np.square(x)/2) / np.sqrt(2*np.pi)

class UpperConfidenceBound:
    """Upper Confidence Bound acquisition function"""
    def __init__(self, kappa=2.576):
        self.kappa = kappa
        
    def __call__(self, X, model, y_best=None):
        """
        Args:
            X: Points to evaluate UCB at
            model: GP model
            y_best: Not used in UCB
        """
        mu, sigma = model.predict(X, return_std=True)
        return mu + self.kappa * sigma

class BayesianTuner:
    def __init__(self, env, config=None):
        """
        Bayesian optimization for database parameter tuning
        
        Args:
            env: Environment with database connection
            config: Configuration dictionary
        """
        self.env = env
        
        # 尝试从不同来源获取配置
        self.config = {}
        
        # 1. 首先尝试使用传入的config
        if config:
            self.config.update(config)
        
        # 2. 然后尝试从config_dict中获取bayesian部分
        if "bayesian" in config_dict:
            self.config.update(config_dict.get("bayesian", {}))
            
        # 3. 最后，尝试从环境的parser.argus中获取
        try:
            if hasattr(env, 'parser') and hasattr(env.parser, 'argus'):
                for key in ['n_init_points', 'acq_func', 'noise_level', 'normalize_y', 
                           'kappa', 'xi', 'random_seed', 'population_size']:
                    if key in env.parser.argus:
                        self.config[key] = env.parser.argus[key]
        except Exception as e:
            print(f"Warning: Could not get config from env.parser.argus: {e}")
        
        # Parse config parameters with defaults
        self.n_init_points = int(self.config.get("n_init_points", 5))
        self.acq_func_name = self.config.get("acq_func", "ei")
        self.noise_level = float(self.config.get("noise_level", 0.1))
        self.normalize_y = self.config.get("normalize_y", "true").lower() == "true"
        self.kappa = float(self.config.get("kappa", 2.576))
        self.xi = float(self.config.get("xi", 0.01))
        self.random_seed = int(self.config.get("random_seed", 42))
        # 添加early_stopping_percentage参数处理
        self.early_stopping_percentage = None
        try:
            if "stopping_throughput_improvement_percentage" in self.config:
                self.early_stopping_percentage = float(self.config.get("stopping_throughput_improvement_percentage"))
            # 如果config中没有，尝试从argus获取
            elif hasattr(env, 'parser') and hasattr(env.parser, 'argus') and 'stopping_throughput_improvement_percentage' in env.parser.argus:
                self.early_stopping_percentage = float(env.parser.argus['stopping_throughput_improvement_percentage'])
        except (ValueError, KeyError, TypeError) as e:
            print(f"Warning: Error parsing stopping_throughput_improvement_percentage: {e}. Using default value 1.")
            self.early_stopping_percentage = 1
        
        # 打印当前使用的配置，便于调试
        print("\n====== Bayesian Optimizer Configuration ======")
        print(f"  n_init_points: {self.n_init_points}")
        print(f"  acq_func: {self.acq_func_name}")
        print(f"  noise_level: {self.noise_level}")
        print(f"  normalize_y: {self.normalize_y}")
        print(f"  kappa: {self.kappa}")
        print(f"  xi: {self.xi}")
        print(f"  random_seed: {self.random_seed}")
        print(f"  early_stopping_percentage: {self.early_stopping_percentage:.2%}" if self.early_stopping_percentage is not None else "  early_stopping_percentage: None")
        print("============================================\n")
        
        # Create random number generator
        self.rng = np.random.RandomState(self.random_seed)
        
        # Set up the acquisition function
        if self.acq_func_name.lower() == 'ei':
            self.acq_func = ExpectedImprovement(xi=self.xi)
        elif self.acq_func_name.lower() == 'ucb':
            self.acq_func = UpperConfidenceBound(kappa=self.kappa)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acq_func_name}")
        
        # Set up the Gaussian Process kernel
        self.kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=self.noise_level)
        
        # The GP regressor
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            normalize_y=self.normalize_y,
            random_state=self.random_seed,
            n_restarts_optimizer=10
        )
        
        # Extract knob information from the environment
        self.parameter_names = list(knob_config.keys())
        self.parameter_bounds = []
        self.parameter_types = []
        self.parameter_scales = []
        
        # 打印参数信息
        print("Parameter information:")
        print(f"  Total parameters: {len(self.parameter_names)}")
        
        for name in self.parameter_names:
            if name in knob_config:
                config = knob_config[name]
                min_val = config.get('min_value', 0)
                max_val = config.get('max_value', 100)
                self.parameter_bounds.append((min_val, max_val))
                
                # Determine if parameter should be integer or float
                if isinstance(min_val, int) and isinstance(max_val, int):
                    self.parameter_types.append('int')
                else:
                    self.parameter_types.append('float')
                    
                # Store the scale for each parameter (used for normalization)
                self.parameter_scales.append(max_val - min_val)
            else:
                # Default bounds if not specified in config
                self.parameter_bounds.append((0, 100))
                self.parameter_types.append('int')
                self.parameter_scales.append(100)
        
        # Convert to numpy array for easier processing
        self.parameter_bounds = np.array(self.parameter_bounds)
        
        # Storage for optimization data
        self.X = []  # Observed parameter configurations
        self.y = []  # Observed performance metrics
        self.best_params = None
        self.best_value = None
        self.count = 0
    
    def _normalize_parameters(self, params):
        """Normalize parameters to [0, 1] range"""
        norm_params = []
        for i, param in enumerate(params):
            min_val, max_val = self.parameter_bounds[i]
            norm_param = (param - min_val) / (max_val - min_val)
            norm_params.append(norm_param)
        return np.array(norm_params)
    
    def _denormalize_parameters(self, norm_params):
        """Convert normalized parameters back to their original range"""
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
        """Suggest the next set of parameters to try"""
        # For initial exploration, generate random parameters
        if self.count < self.n_init_points:
            # Generate random parameters within bounds
            params = []
            for i, (lower, upper) in enumerate(self.parameter_bounds):
                if self.parameter_types[i] == 'int':
                    param = self.rng.randint(lower, upper + 1)
                else:
                    param = self.rng.uniform(lower, upper)
                params.append(param)
            
            self.count += 1
            return np.array(params)
        
        # For later iterations, use Bayesian optimization
        # Generate candidate points (normalized)
        n_candidates = 1000
        candidates = []
        
        for _ in range(n_candidates):
            candidate = np.array([self.rng.uniform(0, 1) for _ in range(len(self.parameter_names))])
            candidates.append(candidate)
        
        candidates = np.array(candidates)
        
        # Normalize observed parameters
        X_normalized = np.array([self._normalize_parameters(x) for x in self.X])
        
        # Fit Gaussian Process model to observed data
        self.gp.fit(X_normalized, self.y)
        
        # Find best observed value
        best_idx = np.argmax(self.y)
        best_value = self.y[best_idx]
        
        # Evaluate acquisition function at all candidates
        acq_values = self.acq_func(candidates, self.gp, best_value)
        
        # Find the candidate with the highest acquisition value
        best_candidate_idx = np.argmax(acq_values)
        best_candidate = candidates[best_candidate_idx]
        
        # Denormalize the parameters
        params = self._denormalize_parameters(best_candidate)
        
        self.count += 1
        return params
    
    def update(self, parameters, performance):
        """Update the model with new observed data
        
        Args:
            parameters: Database parameters that were used
            performance: Performance metric (higher is better)
        """
        self.X.append(parameters)
        self.y.append(performance)
        
        # Update best observed parameters
        if self.best_value is None or performance > self.best_value:
            self.best_value = performance
            self.best_params = parameters
    
    def get_best_parameters(self):
        """Return the best parameters found so far"""
        if self.best_params is None:
            # If no parameters observed yet, return None
            return None
        
        # Create a dictionary of parameter names and values
        param_dict = {}
        for i, name in enumerate(self.parameter_names):
            param_dict[name] = self.best_params[i]
        
        return param_dict
        
    def run_optimization(self, num_trials, early_stopping_percentage=None):
        """Run the complete Bayesian optimization loop
        
        Args:
            num_trials: Number of optimization iterations to run
            early_stopping_percentage: Stop if improvement exceeds this percentage
            
        Returns:
            dict: The best parameters found
            float: The best performance metric achieved
        """
        # 如果函数参数提供了early_stopping_percentage，则使用参数值
        # 否则使用初始化时设置的值
        if early_stopping_percentage is None:
            early_stopping_percentage = self.early_stopping_percentage
            
        best_throughput = 0
        
        print("Starting Bayesian Optimization for database tuning...")
        
        # 记录历史吞吐量值，用于计算改进比例
        throughput_history = []
        
        for iteration in range(num_trials):
            # 获取贝叶斯优化器建议的参数
            params = self.suggest_parameters()
            
            # 应用参数并获取性能指标
            new_state, reward, score, throughput = self.env.step(params, 0, iteration + 1)
            
            # 记录吞吐量
            throughput_history.append(throughput)
            
            # 更新贝叶斯模型
            self.update(params, throughput)
            
            # 打印当前迭代结果
            print(f"[Bayesian] Iteration {iteration+1}/{num_trials}, Throughput: {throughput:.2f}, Reward: {reward:.2f}")
            
            # 记录最佳参数
            if throughput > best_throughput:
                best_throughput = throughput
                print(f"New best throughput: {best_throughput:.2f}")
                
            # 检查是否应该提前停止
            if early_stopping_percentage and iteration > 0 and len(throughput_history) >= 2:
                # 计算相对于初始吞吐量的改进
                initial_throughput = throughput_history[0]
                if initial_throughput > 0:
                    current_improvement = (throughput - initial_throughput) / initial_throughput
                    if current_improvement > early_stopping_percentage:
                        print(f"Reached target improvement: {current_improvement:.2%}. Stopping optimization.")
                        break
        
        # 获取最佳参数
        best_params = self.get_best_parameters()
        
        # 打印最终结果
        print("Bayesian Optimization completed.")
        print(f"Best parameters found: {best_params}")
        print(f"Best throughput: {best_throughput:.2f}")
        
        return best_params, best_throughput 