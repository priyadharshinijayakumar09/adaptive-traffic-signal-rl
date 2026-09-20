import matplotlib.pyplot as plt


# Results from Phase 5
agents = [
    "Random Agent",
    "Q-Learning Agent"
]

average_steps = [
    11.98,
    3.0
]


# Create bar chart
plt.figure(figsize=(8, 5))

plt.bar(agents, average_steps)

plt.xlabel("Agent")
plt.ylabel("Average Steps to Reach Goal")
plt.title("Random Agent vs Q-Learning Agent")

plt.tight_layout()

# Save chart
plt.savefig("results/agent_comparison.png")

plt.show()

print("Comparison chart saved to:")
print("results/agent_comparison.png")