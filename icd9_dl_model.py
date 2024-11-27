import torch.nn as nn
import torch.nn.functional as F
import torch

from cnn import CNN
from d2v_fc import D2VFullyConnectedLayer

class ICD9DeepLearningModel(nn.Module):
    """
    CNN to extract local features
    """

    def __init__(self, word_embedding_size=100,
                 doc_embedding_size=128,
                 d2v_hidden_layer_size=64,
                 dv2_fc_output_size=128,
                 convolution_kernel_sizes=(3, 4, 5),
                 convolution_filter_numbers=(64, 64, 64),
                 d2v_dropout_rate = 0.75,
                 cnn_dropout_rate=0.75,
                 num_labels=1,
                 device='cpu',
                 specific_model=None):
        super(ICD9DeepLearningModel, self).__init__()

        self.device = device

        self.doc_embedding_size = doc_embedding_size
        self.word_embedding_size = word_embedding_size

        self.d2v_fc = D2VFullyConnectedLayer(
            input_size=doc_embedding_size, 
            hidden_size=d2v_hidden_layer_size,
            output_size=dv2_fc_output_size,
            dropout_rate=d2v_dropout_rate,
            device=self.device
        )

        self.cnn = CNN(
            word_embedding_size=word_embedding_size,
            convolution_kernel_sizes=convolution_kernel_sizes,
            convolution_filter_numbers=convolution_filter_numbers,
            dropout_rate=cnn_dropout_rate,
            device=self.device
        )

        total_output_size_from_cnn = 0
        for i in convolution_filter_numbers:
            total_output_size_from_cnn += i

        self.specific_model = None
        if specific_model == "cnn":
            self.specific_model = specific_model
            dv2_fc_output_size = 0
        elif specific_model == "d2v":
            self.specific_model = specific_model
            total_output_size_from_cnn = 0

        self.fc = nn.Linear(total_output_size_from_cnn + dv2_fc_output_size, num_labels)

        


    def forward(self, x):

        x1, x2 = x
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        if self.specific_model == "d2v":
            out = self.d2v_fc(x1)
        elif self.specific_model == "cnn":
            out = self.cnn(x2)
        else:
            x1 = self.d2v_fc(x1)
            x2 = self.cnn(x2)
            out = torch.cat((x1, x2), 1)
        out = self.fc(out)
        return out