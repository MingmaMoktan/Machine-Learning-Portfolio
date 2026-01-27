import gymnasium as gym
import numpy as np
import pickle


map_size = "4x4"

env = gym.make("FrozenLake-v1", map_name=map_size, is_slippery=True, render_mode=None)

# Hyperparameters
learning_rate = 0.05
discount_factor = 0.99
exploration_rate = 1.0
decay_rate = 0.00001

q_table = np.zeros((env.observation_space.n, env.action_space.n))

for episode in range(10000):
    state, _ = env.reset()
    done = False
    
    while not done:
        if np.random.uniform(0,1)<exploration_rate:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
            
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        q_table[state, action] = old_value + learning_rate * (reward + discount_factor* next_max-old_value)
        
        state = next_state
        
    exploration_rate = max(0.01, exploration_rate-decay_rate) # exploration rate decreases and agent picks the decision from the q table
    
with open(f"frozen_lake_{map_size}.pkl", "wb") as f:
    pickle.dump(q_table, f)

env.close()