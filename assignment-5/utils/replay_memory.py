from collections import deque
from random import sample

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size):
        return sample(self.memory, batch_size)


