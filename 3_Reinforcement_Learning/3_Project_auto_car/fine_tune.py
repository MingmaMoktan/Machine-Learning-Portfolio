import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os

# 1. Setting up the paths
# Update these paths if your folder names are slightly different
BASE_DIR = "/home/user/persistent/RL/autonomous_driving_car"
MODEL_TO_LOAD = os.path.join(BASE_DIR, "fine_tuned_best/best_model") # Previous best
SAVE_PATH = os.path.join(BASE_DIR, "best_model")
LOG_PATH = os.path.join(BASE_DIR, "logs_best")

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

# 2. Training environment
# Training env
env = Monitor(gym.make("CarRacing-v2", render_mode="rgb_array"))
# Evaluation env (to check progress)
eval_env = gym.make("CarRacing-v2", render_mode="rgb_array")

# 3. Loading the brain
print(f"Loading existing model from: {MODEL_TO_LOAD}")
# We load the model and connect it to the new training environment
model = PPO.load(MODEL_TO_LOAD, env=env, device="cuda")

# 4. Setup the call back
# This saves a new 'best_model.zip' if the car improves during fine-tuning
eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path=SAVE_PATH,
    log_path=LOG_PATH, 
    eval_freq=10000, # Evaluate every 10k steps
    deterministic=True, 
    render=False
)

# 5. Start Fine tuning
print("Training resumed for 1,000,000 steps. Focus: Consistency!")
# reset_num_timesteps=False keeps the chart line moving forward instead of restarting at 0
model.learn(
    total_timesteps=1000000, 
    callback=eval_callback, 
    reset_num_timesteps=False
)

# 6. Saving the final version
model.save(os.path.join(BASE_DIR, "car_racing_final_pro"))
print("Fine-tuning complete!")