import torch
from torch.utils.data import Dataset

class PhysioDataset(Dataset):
    def __init__(self, data, seq_length=10):
        """
        Initialize the dataset.

        Parameters:
            data (List[dict]): The complete dataset, where each item is a dict with 'frame', 'bvp', 'eda'.
            seq_length (int): The sequence length for the RNN.
        """
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        # The total number of sequences
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, idx):
        # Get sequences of frames, bvp, and eda
        frames = [self.data[idx + i]['frame'] for i in range(self.seq_length)]
        bvp = [self.data[idx + i]['bvp'] for i in range(self.seq_length)]
        eda = [self.data[idx + i]['eda'] for i in range(self.seq_length)]
        
        # Convert lists to tensors
        frames = torch.stack(frames)  # Shape: (seq_length, C, H, W)
        bvp = torch.tensor(bvp, dtype=torch.float32).unsqueeze(-1)  # Shape: (seq_length, 1)
        eda = torch.tensor(eda, dtype=torch.float32)  # Shape: (seq_length,)
        
        return frames, bvp, eda


