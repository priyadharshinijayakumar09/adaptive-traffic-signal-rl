import numpy as np

from environment import SimpleEnvironment


# -----------------------------
# Environment
# -----------------------------
env = SimpleEnvironment()

num_states = 4
num_actions = 2


# -----------------------------
# Q-Learning parameters
# -----------------------------
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


# -----------------------------
# Training
# -----------------------------
for episode in range(episodes):

    state = env.reset()
    done = False

    while not done:

        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            # Explore
            action = np.random.randint(num_actions)
        else:
            # Exploit
            action = np.argmax(q_table[state])

        # Take action
        next_state, reward, done = env.step(action)

        # Find the best future value
        best_future_value = np.max(q_table[next_state])

        # Current Q-value
        current_value = q_table[state, action]

        # Q-learning update
        q_table[state, action] = current_value + learning_rate * (
            reward
            + discount_factor * best_future_value
            - current_value
        )

        # Move to next state
        state = next_state

    # Reduce exploration gradually
    epsilon = max(epsilon_min, epsilon * epsilon_decay)


# -----------------------------
# Results
# -----------------------------
print("Training completed.")

print("\nFinal Q-table:")
print(q_table)

print("\nFinal epsilon:", epsilon)

print("\nLearned policy:")

for state in range(num_states):

    best_action = np.argmax(q_table[state])

    if best_action == 0:
        action_name = "LEFT"
    else:
        action_name = "RIGHT"

    print(f"State {state}: {action_name}")