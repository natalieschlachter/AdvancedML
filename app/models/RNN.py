import torch.nn as nn
import torch.nn.functional as F

class SimpleRNN(nn.Module):
    def __init__(self, cnn_feature_dim=128, bvp_feature_dim=1, hidden_dim=64, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.input_dim = cnn_feature_dim + bvp_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # Predict EDA value

    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_dim)
        out = self.fc(out)  # Shape: (batch_size, seq_length, 1)
        return out.squeeze(-1)  # Shape: (batch_size, seq_length)
