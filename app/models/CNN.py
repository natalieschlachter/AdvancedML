import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, feature_dim=128):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)  # Input channels = 1 for grayscale
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.fc = nn.Linear(64 * 27 * 27, feature_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))  # Output: (batch_size, 16, 118, 118)
        x = F.relu(self.conv2(x))  # Output: (batch_size, 32, 57, 57)
        x = F.relu(self.conv3(x))  # Output: (batch_size, 64, 27, 27)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)             # Output feature vector
        return x
