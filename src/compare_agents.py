import numpy as np

from environment import SimpleEnvironment


# -----------------------------
# Environment
# -----------------------------
env = SimpleEnvironment()

num_states = 4
num_actions = 2

num_episodes = 100


# -----------------------------
# Trained Q-table
# -----------------------------
q_table = np.array([
    [7.28466855, 8.09832610],
    [7.27876971, 8.99908098],
    [8.02962224, 9.99973439],
    [0.0, 0.0]
])


# -----------------------------
# Evaluate Random Agent
# -----------------------------
random_steps = []
random_successes = 0

for episode in range(num_episodes):

    state = env.reset()
    done = False
    steps = 0

    while not done:

        action = np.random.randint(num_actions)

        next_state, reward, done = env.step(action)

        state = next_state
        steps += 1

        # Safety limit
        if steps >= 100:
            break

    random_steps.append(steps)

    if done:
        random_successes += 1


# -----------------------------
# Evaluate Trained Agent
# -----------------------------
trained_steps = []
trained_successes = 0

for episode in range(num_episodes):

    state = env.reset()
    done = False
    steps = 0

    while not done:

        # Choose best learned action
        action = np.argmax(q_table[state])

        next_state, reward, done = env.step(action)

        state = next_state
        steps += 1

        if steps >= 100:
            break

    trained_steps.append(steps)

    if done:
        trained_successes += 1


# -----------------------------
# Results
# -----------------------------
print("Agent Comparison")
print("=" * 40)

print("\nRandom Agent")
print("Average steps:", np.mean(random_steps))
print("Success rate:", random_successes / num_episodes * 100, "%")

print("\nTrained Q-Learning Agent")
print("Average steps:", np.mean(trained_steps))
print("Success rate:", trained_successes / num_episodes * 100, "%")