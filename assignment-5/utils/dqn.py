import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=4, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(self._compute_conv_out(input_shape), 64)
        self.fc2 = nn.Linear(64, num_actions)
        self.dropout = nn.Dropout(p=0.2)

    def _compute_conv_out(self, shape):
        x = torch.zeros(1, *shape)
        x = self.conv1(x)
        x = self.conv2(x)
        return int(np.prod(x.size()))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x