import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from torch.distributions import Categorical
import math
import img_processor

class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()  
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Softmax(),
            nn.Linear(64, 1)  # Output is a scalar value
        )
    
    def forward(self, x):
        return self.network(x)