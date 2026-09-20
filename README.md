# 🚦 Adaptive Traffic Signal Control using Reinforcement Learning

An end-to-end Reinforcement Learning project that demonstrates how an agent can learn an optimal traffic-signal control policy using **Q-Learning** and **Deep Q-Networks (DQN)**.

## 📌 Project Overview

The project starts with a simple reinforcement learning environment where the agent must learn to reach a target state.

Two reinforcement learning approaches are implemented and compared:

- Q-Learning
- Deep Q-Network (DQN)

The trained agents are evaluated based on:

- Average steps required to reach the goal
- Success rate
- Total reward

A Streamlit application provides an interactive demonstration of the trained agents.

## 🧠 Reinforcement Learning Concepts

The project covers:

- Agent
- Environment
- State
- Action
- Reward
- Policy
- Q-table
- Q-values
- Exploration vs exploitation
- Epsilon-greedy strategy
- Bellman update
- Neural-network-based Q-value approximation

## 🤖 Algorithms

### Q-Learning

A tabular reinforcement learning algorithm where Q-values are stored in a Q-table.

The agent updates its Q-value using:

```text
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]