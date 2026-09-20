import sys
from pathlib import Path

import streamlit as st
import numpy as np
import torch


# ==========================================
# Project Path
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.environment import SimpleEnvironment


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Adaptive Traffic Signal RL",
    page_icon="🚦",
    layout="centered"
)


# ==========================================
# Title
# ==========================================

st.title("🚦 Adaptive Traffic Signal Control")
st.subheader("Reinforcement Learning Demonstration")

st.write(
    "Compare Q-Learning and DQN agents in a simple "
    "traffic signal control environment."
)


# ==========================================
# Agent Selection
# ==========================================

agent_type = st.selectbox(
    "Select Agent",
    ["Q-Learning", "DQN"]
)


# ==========================================
# Run Simulation
# ==========================================

if st.button("Run Simulation"):

    env = SimpleEnvironment()

    state = env.reset()
    done = False

    steps = 0
    total_reward = 0

    st.write("### Simulation")

    while not done:

        # ----------------------------------
        # Q-Learning
        # ----------------------------------

        if agent_type == "Q-Learning":

            if state < 3:
                action = 1
            else:
                action = 0

        # ----------------------------------
        # DQN
        # ----------------------------------

        else:

            from src.dqn_agent import DQN

            model = DQN(
                state_size=4,
                action_size=2
            )

            model_path = PROJECT_ROOT / "results" / "dqn_model.pth"

            model.load_state_dict(
                torch.load(
                    model_path,
                    weights_only=True
                )
            )

            model.eval()

            state_tensor = torch.zeros(4)
            state_tensor[state] = 1.0

            with torch.no_grad():

                q_values = model(state_tensor)

            action = torch.argmax(q_values).item()

        # ----------------------------------
        # Environment Step
        # ----------------------------------

        next_state, reward, done = env.step(action)

        steps += 1
        total_reward += reward

        action_name = (
            "LEFT"
            if action == 0
            else "RIGHT"
        )

        st.write(
            f"Step {steps}: "
            f"State {state} → "
            f"Action **{action_name}** → "
            f"State {next_state} → "
            f"Reward **{reward}**"
        )

        state = next_state

    # ======================================
    # Final Results
    # ======================================

    st.success("Simulation completed!")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Steps",
        steps
    )

    col2.metric(
        "Total Reward",
        total_reward
    )

    col3.metric(
        "Final State",
        state
    )