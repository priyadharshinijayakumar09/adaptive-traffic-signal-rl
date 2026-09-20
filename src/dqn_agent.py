import torch
import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )

    def forward(self, state):
        return self.network(state)


# Test the DQN only when this file is run directly
if __name__ == "__main__":

    state_size = 4
    action_size = 2

    model = DQN(state_size, action_size)

    print("DQN model created successfully.")
    print(model)

    state = torch.tensor(
        [0.0, 0.0, 0.0, 0.0]
    )

    q_values = model(state)

    print("\nInput state:")
    print(state)

    print("\nQ-values:")
    print(q_values)

    print("\nBest action:")
    print(torch.argmax(q_values).item())