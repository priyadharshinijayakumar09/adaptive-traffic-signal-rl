import numpy as np
import matplotlib.pyplot as plt

from environment import SimpleEnvironment


# -----------------------------
# Environment
# -----------------------------
env = SimpleEnvironment()

num_states = 4
num_actions = 2

learning_rate = 0.1
discount_factor = 0.9

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 100


# -----------------------------
# Q-table
# -----------------------------
q_table = np.zeros((num_states, num_actions))


# Store steps for each episode
episode_steps = []


# -----------------------------
# Training
# -----------------------------
for episode in range(episodes):

    state = env.reset()
    done = False
    steps = 0

    while not done:

        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(q_table[state])

        # Take action
        next_state, reward, done = env.step(action)

        # Q-learning update
        best_future_value = np.max(q_table[next_state])
        current_value = q_table[state, action]

        q_table[state, action] = current_value + learning_rate * (
            reward
            + discount_factor * best_future_value
            - current_value
        )

        state = next_state
        steps += 1

        # Safety limit
        if steps >= 100:
            break

    episode_steps.append(steps)

    # Reduce exploration
    epsilon = max(epsilon_min, epsilon * epsilon_decay)


# -----------------------------
# Calculate moving average
# -----------------------------
window = 10

moving_average = np.convolve(
    episode_steps,
    np.ones(window) / window,
    mode="valid"
)


# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 5))

plt.plot(
    range(1, episodes + 1),
    episode_steps,
    alpha=0.4,
    label="Steps per Episode"
)

plt.plot(
    range(window, episodes + 1),
    moving_average,
    linewidth=2,
    label="10-Episode Moving Average"
)

plt.xlabel("Episode")
plt.ylabel("Steps to Reach Goal")
plt.title("Q-Learning Training Progress")

plt.legend()
plt.grid(True)

plt.tight_layout()

# Save result
plt.savefig("results/learning_curve.png")

plt.show()


print("Training completed.")
print("Learning curve saved to:")
print("results/learning_curve.png")