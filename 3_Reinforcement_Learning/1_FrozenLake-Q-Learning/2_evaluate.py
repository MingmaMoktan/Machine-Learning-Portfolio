import gymnasium as gym
import numpy as np
import pickle
import time

MAP_NAME = "4x4" 
PICKLE_FILE = f"frozen_lake_{MAP_NAME}.pkl"

def run_evaluation():
    try:
        with open(PICKLE_FILE, "rb") as f:
            q_table = pickle.load(f)
        print(f"Successfully loaded Q-table for {MAP_NAME}")
    except FileNotFoundError:
        print(f"Error: {PICKLE_FILE} not found. Run your training script first!")
        return
    env = gym.make("FrozenLake-v1", map_name=MAP_NAME, is_slippery=True, render_mode="human")
    
    for episode in range(1, 4):
        state, _ = env.reset()
        done = False
        truncated = False
        print(f"Starting Episode {episode}...")

        while not (done or truncated):
            action = np.argmax(q_table[state])
            state, reward, done, truncated, _ = env.step(action)
            time.sleep(0.2)
        if reward == 1:
            print(f"Result: Episode {episode} - GOAL REACHED! 🎉")
        else:
            print(f"Result: Episode {episode} - FELL IN HOLE ❄️")
        
        time.sleep(1)

    env.close()

if __name__ == "__main__":
    run_evaluation()