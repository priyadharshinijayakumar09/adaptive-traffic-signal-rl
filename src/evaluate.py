import numpy as np

from environment import SimpleEnvironment


# -----------------------------
# Environment
# -----------------------------
env = SimpleEnvironment()

num_states = 4
num_actions = 2


# -----------------------------
# Q-table from training
# -----------------------------
q_table = np.array([
    [7.28466855, 8.09832610],
    [7.27876971, 8.99908098],
    [8.02962224, 9.99973439],
    [0.0, 0.0]
])


# -----------------------------
# Evaluate trained agent
# -----------------------------
state = env.reset()

steps = 0
done = False

print("Starting evaluation...")
print("Initial state:", state)

while not done:

    # Choose the best learned action
    action = np.argmax(q_table[state])

    # Take action
    next_state, reward, done = env.step(action)

    steps += 1

    print(
        f"Step {steps}: "
        f"State {state} -> "
        f"Action {action} -> "
        f"Next State {next_state} -> "
        f"Reward {reward}"
    )

    state = next_state


print("\nEvaluation completed.")
print("Steps required:", steps)
print("Final state:", state)