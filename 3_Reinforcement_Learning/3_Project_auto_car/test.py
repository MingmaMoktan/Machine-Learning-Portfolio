import gymnasium as gym
from stable_baselines3 import PPO

# 1. Load the environment
# Use 'human' render_mode IF you are on a local PC with a monitor.
# Use 'rgb_array' if you are just testing logic on the server.
env = gym.make("CarRacing-v2", render_mode="human") 

# 2. Load the Best Model
model = PPO.load("./best_model/best_model.zip")

# 3. Run Test Episodes
episodes = 3
for ep in range(1, episodes + 1):
    obs, info = env.reset()
    done = truncated = False
    score = 0
    
    while not (done or truncated):
        # Predict the action using the trained brain
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        score += reward
        env.render() # This opens the window on a local PC
        
    print(f"Test Episode {ep} | Final Score: {score:.2f}")

env.close()