import torch

from environment import SimpleEnvironment
from dqn_agent import DQN


# -----------------------------
# Environment
# -----------------------------
env = SimpleEnvironment()

state_size = 4
action_size = 2


# -----------------------------
# Create DQN model
# -----------------------------
model = DQN(state_size, action_size)

# Load trained model
model.load_state_dict(
    torch.load(
        "results/dqn_model.pth",
        weights_only=True
    )
)

model.eval()


# -----------------------------
# Convert state to one-hot
# -----------------------------
def state_to_tensor(state):
    state_vector = torch.zeros(4)
    state_vector[state] = 1.0
    return state_vector


# -----------------------------
# Evaluate DQN
# -----------------------------
state = env.reset()

done = False
steps = 0
total_reward = 0


print("Starting DQN evaluation...")
print("Initial state:", state)


while not done:

    state_tensor = state_to_tensor(state)

    with torch.no_grad():
        q_values = model(state_tensor)

    action = torch.argmax(q_values).item()

    next_state, reward, done = env.step(action)

    steps += 1
    total_reward += reward

    action_name = "LEFT" if action == 0 else "RIGHT"

    print(
        f"Step {steps}: "
        f"State {state} -> "
        f"Action {action_name} -> "
        f"Next State {next_state} -> "
        f"Reward {reward}"
    )

    state = next_state


print("\nDQN evaluation completed.")
print("Steps required:", steps)
print("Total reward:", total_reward)
print("Final state:", state)