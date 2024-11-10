import torch.nn as nn
import torch.nn.functional as F
import torch

class D2VFullyConnectedLayer(nn.Module):
    """
    CNN to extract local features
    """

    def __init__(self, input_size=128, hidden_size=64, output_size=128, dropout_rate=0.75, device='cpu'):
        super(D2VFullyConnectedLayer, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.to(self.device)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x