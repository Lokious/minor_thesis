import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import random
import matplotlib.pyplot as plt

from simulated_data import return_simulated_dataset

class Encoder(nn.Module):
    def __init__(self, input_size=512, hidden_size=128, num_layers=2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x):
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        outputs, (hidden, cell) = self.lstm(x)
        return (hidden, cell)

class Decoder(nn.Module):
    def __init__(
        self, input_size=512, hidden_size=128, output_size=512, num_layers=2
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        output, (hidden, cell) = self.lstm(x, hidden)
        prediction = self.fc(output)
        return prediction, (hidden, cell)

















def main():
    #generate simulated  dataset
    simulated_dataset = return_simulated_dataset()

    #set device: cup or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    main()
