class SimpleEnvironment:
    def __init__(self):
        self.goal = 3
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):

        # Action 0 = LEFT
        # Action 1 = RIGHT

        if action == 1:
            self.state += 1
        elif action == 0:
            self.state -= 1

        # Keep state inside the environment
        self.state = max(0, min(self.state, self.goal))

        # Reward
        if self.state == self.goal:
            reward = 10
            done = True
        else:
            reward = 0
            done = False

        return self.state, reward, done