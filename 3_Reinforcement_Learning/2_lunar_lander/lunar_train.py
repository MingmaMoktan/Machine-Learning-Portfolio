import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os

env = gym.make("LunarLander-v2")
env = Monitor(env)

configs = [
    {
        'learning_rate': 1e-3,
        'batch_size': 32,
        'buffer_size': 50000,
        'gamma': 0.99,
        'train_freq': 1,           # Train EVERY step (4x faster!)
        'gradient_steps': 1,       # 1 gradient update per step
        'learning_starts': 5000,   # Shorter warmup
        'target_update_interval': 500
    },
    {
        'learning_rate': 5e-4,
        'batch_size': 64,
        'buffer_size': 50000,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
        'learning_starts': 5000,
        'target_update_interval': 500
    },
    {
        'learning_rate': 1e-3,
        'batch_size': 64,
        'buffer_size': 50000,
        'gamma': 0.995,
        'train_freq': 1,
        'gradient_steps': 1,
        'learning_starts': 5000,
        'target_update_interval': 250
    },
    {
        'learning_rate': 2e-4,
        'batch_size': 128,
        'buffer_size': 100000,
        'gamma': 0.99,
        'train_freq': 1,
        'gradient_steps': 1,
        'learning_starts': 5000,
        'target_update_interval': 1000
    }
]

os.makedirs("models", exist_ok=True)
print("Training")

for i, config in enumerate(configs):
    config_folder = f"./models/config_{i+1}/"
    os.makedirs(config_folder, exist_ok=True)
    
    print(f"\n--- Starting Config {i+1}/4 ---")

    # Creates fresh env for evaluation   
    eval_env = Monitor(gym.make("LunarLander-v2"))
    
    # Initialize the model with the CURRENT config
    model = DQN("MlpPolicy", env, verbose=1, **config) 
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=config_folder, 
        log_path=config_folder,             
        eval_freq=10000,
        verbose=1
    )
    
    model.learn(total_timesteps=75000, callback=eval_callback)

env.close()