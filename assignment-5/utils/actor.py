import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, num_actions):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=8, stride=8, padding=0)
        self.fc1 = nn.Linear(8 * 9 * 9, 64)
        self.fc2 = nn.Linear(64, num_actions)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.pool1(x)
        x = x.view(-1, 8 * 9 * 9)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x