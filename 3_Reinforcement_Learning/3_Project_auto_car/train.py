import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os

# Create directory for the best model
save_path = "./best_model/"
os.makedirs(save_path, exist_ok=True)

# 1. Setup Environments
env = Monitor(gym.make("CarRacing-v2", render_mode="rgb_array"))
eval_env = gym.make("CarRacing-v2", render_mode="rgb_array")

# 2. Setup Callback to save the BEST model automatically
eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path=save_path,
    log_path='./logs/', 
    eval_freq=5000, # Check every 5000 steps
    deterministic=True, 
    render=False
)

# 3. Initialize and Train
model = PPO("CnnPolicy", env, verbose=1, device="cuda")
print("Training started. Best model will be saved to './best_model/best_model.zip'")
model.learn(total_timesteps=1000000, callback=eval_callback)

env.close()