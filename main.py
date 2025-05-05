import sys
import time
import numpy as np
import torch
import json
import configparser
import os
from environment import Database, Environment
from configs import parse_args
from get_workload_from_file import get_workload_from_file
from registry import OptimizerRegistry

# Import optimizer implementations to register them
import optimizers

if __name__ == "__main__":
    # Parse arguments
    argus = parse_args()

    # Prepare training workloads
    training_workloads = []
    workload = get_workload_from_file(argus["workload_file_path"])
    argus["workload"] = workload
    
    # Set PyTorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize database and environment
    db = Database(argus)  # connector knobs metric
    env = Environment(db, argus)
    
    # 确保必要的目录存在
    os.makedirs("training-results", exist_ok=True)
    os.makedirs("saved_model_weights", exist_ok=True)

    # Read config file to pass to optimizer
    config_parser = configparser.ConfigParser()
    config_parser.read("config_1.ini")
    
    # Create config dictionary from config parser
    config_dict = {}
    for section in config_parser.sections():
        config_dict[section.lower()] = {}
        for key, value in config_parser[section].items():
            # Try to evaluate the value as a Python expression (for dict values)
            try:
                if value.startswith('{') and value.endswith('}'):
                    config_dict[section.lower()][key] = eval(value)
                else:
                    config_dict[section.lower()][key] = value
            except:
                config_dict[section.lower()][key] = value

    # Get optimizer type from arguments or config
    optimizer_type = argus.get('optimizer_type', 'adam')  # Default to adam if not specified
    
    try:
        # Get number of trials from arguments
        num_trials = int(argus.get('num_trial', 10))
        
        # Print available optimizers
        print("\n------ Available Optimizers ------")
        available_optimizers = list(OptimizerRegistry.list_optimizers().keys())
        print(f"Available optimizers: {', '.join(available_optimizers)}")
        print(f"Selected optimizer: {optimizer_type}")
        
        # Initialize optimizer from registry
        optimizer_kwargs = {
            'env': env,
            'learning_rate': float(argus['learning_rate']),
            'train_min_size': int(argus['train_min_size']),
            'size_mem': int(argus['maxlen_mem']), 
            'size_predict_mem': int(argus['maxlen_predict_mem']),
            'config_dict': config_dict
        }
        
        try:
            optimizer = OptimizerRegistry.get_optimizer(optimizer_type, **optimizer_kwargs)
            
            # Run optimization
            print(f"\nRunning optimization with {optimizer_type} optimizer for {num_trials} trials...\n")
            best_params, best_throughput = optimizer.optimize(num_trials=num_trials)
            
            # Print optimization summary
            print(f"\n------ {optimizer_type.capitalize()} Optimization Summary ------")
            print(f"Best throughput achieved: {best_throughput:.2f}")
            
            print("\n------ Optimization Completed ------\n")
            
        except ValueError as e:
            # Handle case when optimizer is not found in registry
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during optimization: {e}")

    # TODO: 训练predict
    # sample_times = 2
    # for i in range(sample_times):
    #     training_workloads.append(np.random.choice(workload, np.random.randint(len(workload)), replace=False, p=None))
    # X = []
    # Y = []
    # for w in training_workloads:
    #     vec = env.parser.get_workload_encoding(w)
    #     X.append(vec.flatten())
    #     state0 = env.db.fetch_internal_metrics()
    #     env.preheat()
    #     state1 = env.db.fetch_internal_metrics()
    #     Y.append(state1 - state0)
    # X = np.array(X)
    # Y = np.array(Y)
    # env.parser.estimator.fit(X, Y, batch_size=50, epochs=predictor_epoch)

    # TODO save&load model e.g. env.parser.estimator.save_weights(path)
    # env.parser.estimator.save_weights(filepath=path)
    # env.parser.estimator.load_weights(filepath=path)
