

# Project Report: Q-Learning on Frozen Lake (4x4 vs 8x8)

## 1. Project Overview
The goal of this project is to train an autonomous agent to navigate a "Frozen Lake" grid. The agent must find a gift (Goal) while avoiding holes, all while dealing with "slippery ice" that makes its movements unpredictable.

## 2. The Decision Logic (Exploration vs. Exploitation)
To learn effectively, the agent uses an **Epsilon-Greedy Strategy**. This is controlled by the following logic:

```python
if np.random.uniform(0, 1) < exploration_rate:
    action = env.action_space.sample() # Explore: Random move
else:
    action = np.argmax(q_table[state]) # Exploit: Best move from memory

```

* **Early Training:** The `exploration_rate` is high (1.0), forcing the agent to move randomly and discover the map.
* **The Decay:** As the agent plays more episodes, the `exploration_rate` decreases by a `decay_rate`. This forces the agent to stop "guessing" and start trusting its learned values.
* **The Result:** By the end of training, the agent follows the Max Q-Value, choosing the path it has proven to be the most successful.

## 3. The Hyperparameters

These settings act as the "personality" of the agent:

* **Learning Rate ():** Controls how quickly the agent accepts new information. A low value (5%) is used to prevent the agent from overreacting to "lucky" moves caused by the slippery ice.
* **Discount Factor ():** Determines the importance of future rewards. A value of 0.99 ensures the agent prioritizes the long-term goal over immediate safety.
* **Decay Rate ():** Crucial for the 8x8 map, this slow decay keeps the agent curious long enough to explore all 64 possible squares.

## 4. The Learning Engine (Bellman Equation)

Learning happens through the Q-Learning update rule. This formula calculates the difference between the agent's expectation and the actual outcome:

$$Q(s, a) \leftarrow Q(s, a) + \alpha [R + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

**How it works in the code:**

* **next_state ():** Where the agent landed.
* In 4x4 map: 0–15 (16 states)
* In 8x8 map: 0–63 (64 states)


* **reward ():** 1 for success (Goal), 0 for failure/path.
* **terminated:** Boolean flag; ends the episode if the agent falls in a hole or reaches the goal.
* **truncated:** Boolean flag; ends the episode if the step limit is reached.
* **episodes:** One complete cycle of training from start to finish.
* **Steps:** The individual movements (Up, Down, Left, Right).

**Key Concepts:**

* **Old Value:** What the agent previously thought the move was worth.
* **Next Max:** The best possible score the agent can get from the square it just landed on.
* **Reward Propagation:** Every time the agent hits the goal, the Bellman equation assigns a value. Over thousands of episodes, these values "leak" backward through the Q-table, creating a visible path from the Start to the Goal.

## 5. Map Comparison: 4x4 vs. 8x8

* **4x4 Map:** 16 states. Learning is fast because the reward signal from the goal only has to travel back a few steps.
* **8x8 Map:** 64 states. Significantly harder. The agent needs 50,000 episodes and a much slower decay rate because the odds of hitting the goal by accident are much lower (Sparse Reward Problem).

## 6. Persistence (The Pickle File)

Once training is complete, the `q_table` (the agent's brain) is saved as a `.pkl` file. This allows the model to be evaluated instantly without retraining, demonstrating that the agent has successfully converged on an optimal policy for the environment.