import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()  
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Softmax(dim=-1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.network(x)