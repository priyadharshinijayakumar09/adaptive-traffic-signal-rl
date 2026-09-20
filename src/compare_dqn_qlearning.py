import numpy as np
import torch

from environment import SimpleEnvironment
from dqn_agent import DQN


# ==========================================
# Configuration
# ==========================================

NUM_EPISODES = 100

state_size = 4
action_size = 2


# ==========================================
# Q-Learning Agent
# ==========================================

def train_q_learning():

    env = SimpleEnvironment()

    q_table = np.zeros((state_size, action_size))

    learning_rate = 0.1
    discount_factor = 0.9

    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01

    for _ in range(500):

        state = env.reset()
        done = False

        while not done:

            if np.random.random() < epsilon:
                action = np.random.randint(action_size)
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done = env.step(action)

            best_future_value = np.max(q_table[next_state])

            q_table[state, action] += learning_rate * (
                reward
                + discount_factor * best_future_value
                - q_table[state, action]
            )

            state = next_state

        epsilon = max(
            epsilon_min,
            epsilon * epsilon_decay
        )

    return q_table


# ==========================================
# Evaluate Q-Learning
# ==========================================

def evaluate_q_learning(q_table):

    env = SimpleEnvironment()

    steps_list = []
    success_count = 0

    for _ in range(NUM_EPISODES):

        state = env.reset()
        done = False
        steps = 0

        while not done:

            action = np.argmax(q_table[state])

            next_state, reward, done = env.step(action)

            state = next_state
            steps += 1

        steps_list.append(steps)

        if state == env.goal:
            success_count += 1

    return np.mean(steps_list), success_count / NUM_EPISODES * 100


# ==========================================
# DQN
# ==========================================

def state_to_tensor(state):

    state_vector = torch.zeros(state_size)
    state_vector[state] = 1.0

    return state_vector


# ==========================================
# Evaluate DQN
# ==========================================

def evaluate_dqn():

    env = SimpleEnvironment()

    model = DQN(state_size, action_size)

    model.load_state_dict(
        torch.load(
            "results/dqn_model.pth",
            weights_only=True
        )
    )

    model.eval()

    steps_list = []
    success_count = 0

    for _ in range(NUM_EPISODES):

        state = env.reset()
        done = False
        steps = 0

        while not done:

            state_tensor = state_to_tensor(state)

            with torch.no_grad():
                q_values = model(state_tensor)

            action = torch.argmax(q_values).item()

            next_state, reward, done = env.step(action)

            state = next_state
            steps += 1

        steps_list.append(steps)

        if state == env.goal:
            success_count += 1

    return np.mean(steps_list), success_count / NUM_EPISODES * 100


# ==========================================
# Run Comparison
# ==========================================

print("\nTraining Q-Learning agent...")

q_table = train_q_learning()

q_steps, q_success = evaluate_q_learning(q_table)

dqn_steps, dqn_success = evaluate_dqn()


# ==========================================
# Results
# ==========================================

print("\n==========================================")
print("DQN vs Q-Learning Comparison")
print("==========================================")

print("\nQ-Learning")
print("Average steps:", q_steps)
print("Success rate:", q_success, "%")

print("\nDQN")
print("Average steps:", dqn_steps)
print("Success rate:", dqn_success, "%")

print("\n==========================================")