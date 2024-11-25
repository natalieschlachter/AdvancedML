import torch.nn as nn
import torch.nn.functional as F

class CombinedModel(nn.Module):
    def __init__(self, cnn, rnn):
        super(CombinedModel, self).__init__()
        self.cnn = cnn
        self.rnn = rnn

    def forward(self, frames, bvp):
        batch_size, seq_length, C, H, W = frames.size()
        frames = frames.view(batch_size * seq_length, C, H, W)  # Flatten batch and seq dimensions
        cnn_features = self.cnn(frames)  # Shape: (batch_size * seq_length, cnn_feature_dim)
        
        # Reshape back to (batch_size, seq_length, cnn_feature_dim)
        cnn_features = cnn_features.view(batch_size, seq_length, -1)
        
        # Concatenate CNN features with BVP values
        bvp = bvp.view(batch_size, seq_length, -1)  # Ensure shape is correct
        rnn_input = torch.cat((cnn_features, bvp), dim=2)  # Shape: (batch_size, seq_length, input_dim)
        
        # Pass through RNN
        output = self.rnn(rnn_input)  # Shape: (batch_size, seq_length)
        return output
