import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment import SimpleEnvironment
from dqn_agent import DQN


# -----------------------------
# Convert state to one-hot tensor
# -----------------------------
def state_to_tensor(state):
    state_vector = torch.zeros(4)
    state_vector[state] = 1.0
    return state_vector


# -----------------------------
# Environment
# -----------------------------
env = SimpleEnvironment()

state_size = 4
action_size = 2


# -----------------------------
# DQN model
# -----------------------------
model = DQN(state_size, action_size)

loss_function = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# -----------------------------
# Training parameters
# -----------------------------
episodes = 500

discount_factor = 0.9

epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

episode_rewards = []


# -----------------------------
# Training
# -----------------------------
for episode in range(episodes):

    state = env.reset()
    done = False

    total_reward = 0
    steps = 0

    while not done:

        # -----------------------------
        # Epsilon-greedy action selection
        # -----------------------------
        if np.random.random() < epsilon:

            # Explore
            action = np.random.randint(action_size)

        else:

            # Exploit
            state_tensor = state_to_tensor(state)

            with torch.no_grad():
                q_values = model(state_tensor)

            action = torch.argmax(q_values).item()


        # -----------------------------
        # Take action
        # -----------------------------
        next_state, reward, done = env.step(action)


        # -----------------------------
        # Current Q-value
        # -----------------------------
        state_tensor = state_to_tensor(state)

        q_values = model(state_tensor)

        current_q = q_values[action]


        # -----------------------------
        # Target Q-value
        # -----------------------------
        next_state_tensor = state_to_tensor(next_state)

        with torch.no_grad():

            if done:

                target_q = torch.tensor(
                    float(reward),
                    dtype=torch.float32
                )

            else:

                next_q_values = model(next_state_tensor)

                target_q = torch.tensor(
                    float(reward),
                    dtype=torch.float32
                ) + discount_factor * torch.max(
                    next_q_values
                )


        # -----------------------------
        # Calculate loss
        # -----------------------------
        loss = loss_function(
            current_q,
            target_q
        )


        # -----------------------------
        # Backpropagation
        # -----------------------------
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        # -----------------------------
        # Move to next state
        # -----------------------------
        state = next_state

        total_reward += reward
        steps += 1


        # -----------------------------
        # Safety limit
        # -----------------------------
        if steps >= 100:
            break


    # Store episode reward
    episode_rewards.append(total_reward)


    # -----------------------------
    # Reduce exploration
    # -----------------------------
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


    # -----------------------------
    # Print progress
    # -----------------------------
    if (episode + 1) % 50 == 0:

        average_reward = np.mean(
            episode_rewards[-50:]
        )

        print(
            f"Episode {episode + 1}/500 | "
            f"Average Reward: {average_reward:.2f} | "
            f"Epsilon: {epsilon:.3f}"
        )


# -----------------------------
# Training completed
# -----------------------------
print("\nDQN training completed.")
torch.save(
    model.state_dict(),
    "results/dqn_model.pth"
)

print("DQN model saved to:")
print("results/dqn_model.pth")