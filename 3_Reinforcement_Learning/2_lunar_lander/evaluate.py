import gymnasium as gym
from stable_baselines3 import DQN

# Model Path
MODEL_PATH = "models/config_2/best_model.zip" 

# Env
env = gym.make("LunarLander-v2", render_mode="human")

# Load model
model = DQN.load(MODEL_PATH)

for ep in range(5): # Test 5 times to get an average
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        total_reward += reward
        done = term or trunc
    
    print(f"Model: {MODEL_PATH} | Episode {ep+1} | Score: {total_reward:.2f}")

env.close()