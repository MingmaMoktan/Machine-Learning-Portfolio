# 🚀 Machine Learning & AI Portfolio

Welcome to my Machine Learning journey! This repository documents my transition from foundational concepts to advanced autonomous agents, featuring projects in Supervised, Unsupervised, and Reinforcement Learning.

<p align="center">
  <code><img height="30" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg"></code>
  <code><img height="30" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg"></code>
  <code><img height="30" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pandas/pandas-original.svg"></code>
  <code><img height="30" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/postgresql/postgresql-original.svg"></code>
  <code><img height="30" src="Assets/Apache_airflow.png"></code>
  <code><img height="30" src="Assets/Apache_kafka.png"></code>
  <code><img height="30" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/apachespark/apachespark-original.svg"></code>
  <code><img height="30" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/docker/docker-original.svg"></code>
  <code><img height="30" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/linux/linux-original.svg"></code>
  <code><img height="30" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/django/django-plain.svg"></code>
  <code><img height="30" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/mongodb/mongodb-original.svg"></code>
  <code><img height="30" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/html5/html5-original-wordmark.svg"></code>
  <code><img height="30" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/css3/css3-original-wordmark.svg"></code>
  <code><img height="30" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/javascript/javascript-original.svg"></code>
  <code><img height="30" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/mysql/mysql-original-wordmark.svg"></code>
  <code><img height="30" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/azure/azure-original.svg"></code>
</p>

---

## 📂 Portfolio Roadmap

### 🤖 01. Reinforcement Learning
* **[Frozen Lake 8x8 Q-Learning](./03-Reinforcement-Learning/Frozen-Lake-Q-Learning/)**
  * **Objective:** Autonomous navigation on a slippery 8x8 grid.
  * **Algorithm:** Q-Learning with Epsilon-Greedy strategy.
  * **Status:** ✅ Complete

### 📈 02. Supervised Learning
* *Project Coming Soon*

### 🧩 03. Unsupervised Learning
* *Project Coming Soon*

---

## ❄️ Project Spotlight: Frozen Lake (4x4 vs 8x8)

### 1. Project Overview
The goal is to train an agent to navigate a "Frozen Lake" grid. The agent must find a Goal while avoiding holes, navigating "slippery ice" where actions only have a 1/3 probability of moving in the intended direction.

### 2. Decision Logic (Exploration vs. Exploitation)
The agent utilizes an **Epsilon-Greedy Strategy** to discover the environment:
- **Explore:** Move randomly to discover rewards.
- **Exploit:** Trust learned Q-values to take the best path.
- **Decay:** Exploration decreases over time as the agent gains confidence.

### 3. Hyperparameters
- **Learning Rate ($\alpha = 0.05$):** Prevents overreacting to "lucky" moves on slippery ice.
- **Discount Factor ($\gamma = 0.99$):** Essential for 8x8 maps to ensure the reward signal propagates back to the start.
- **Decay Rate ($0.00001$):** Keeps the agent curious long enough to explore all 64 states.

### 4. The Bellman Equation Update
$$Q(s, a) \leftarrow Q(s, a) + \alpha [R + \gamma \max Q(s', a') - Q(s, a)]$$

This formula allows the agent to update its expectations based on the reward ($R$) and the future potential value of the next state ($s'$).

### 5. Why 8x8 is harder?
The 8x8 map introduces a **Sparse Reward Problem**. In a 4x4 grid, a random agent hits the goal easily. In an 8x8 grid (64 states), the reward is so far away that the agent requires 50,000 episodes and a high $\gamma$ to "bridge the gap" and find a successful path.

---

## 📫 Connect with me:
<p align="left">
  <a href="https://dm3339.pythonanywhere.com/about/" target="_blank">
    <img src="https://img.shields.io/badge/Portfolio-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Website">
  </a>
  <a href="YOUR_LINKEDIN_URL" target="_blank">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
  <a href="mailto:YOUR_EMAIL@EXAMPLE.COM">
    <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email">
  </a>
</p>