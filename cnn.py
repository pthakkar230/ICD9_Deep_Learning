import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN(nn.Module):
    """
    CNN to extract local features
    """

    def __init__(self,
                 word_embedding_size=100,
                 convolution_kernel_sizes=(3, 4, 5),
                 convolution_filter_numbers=(64, 64, 64),
                 dropout_rate=0.75):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=word_embedding_size, 
            out_channels=convolution_filter_numbers[0],
            kernel_size=convolution_kernel_sizes[0],
        )

        self.conv2 = nn.Conv1d(
            in_channels=word_embedding_size, 
            out_channels=convolution_filter_numbers[1],
            kernel_size=convolution_kernel_sizes[1]
        )

        self.conv3 = nn.Conv1d(
            in_channels=word_embedding_size, 
            out_channels=convolution_filter_numbers[2],
            kernel_size=convolution_kernel_sizes[2]
        )

        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, X):

        out1 = self.pool(self.relu(self.conv1(X))).squeeze(2)
        out2 = self.pool(self.relu(self.conv2(X))).squeeze(2)
        out3 = self.pool(self.relu(self.conv3(X))).squeeze(2)

        out = torch.cat((out1, out2, out3), 1)
        out = self.dropout(out)
        return out





    