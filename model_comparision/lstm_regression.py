import dill
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from prettytable import PrettyTable
from torch.nn import functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import torch.optim as optim
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchmetrics.classification import BinaryMatthewsCorrCoef
def normalization(dataset:torch.tensor)->torch.tensor:

    # normalization to (1,2)

    print(dataset.shape)
    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    scaler = StandardScaler()
    timeseries_tensor = []

    dataset = torch.permute(dataset, (2, 1, 0))
    #print(dataset.shape)
    tensors = list(dataset)
    # print(tensors)
    for feature_df in tensors:
        # print("size of the input of scaler")
        # print(feature_df.shape)
        scaler = scaler.fit(feature_df)
        # normalize the dataset and print
        standardized = scaler.transform(feature_df)
        #print(standardized.shape)
        standardized = torch.stack([torch.tensor(s).float() for s in standardized])
        timeseries_tensor.append(standardized)
        # put time step at the first dimention
    timeseries_tensor = torch.stack(timeseries_tensor)
    # print("before permute")
    # print(timeseries_tensor.shape)
    dataset = torch.permute(timeseries_tensor, (2, 1, 0))  # (time step, number of sequences, inputsize(3))

    # print("after normalize")
    print(dataset.shape)
    return dataset,scaler

class LSTM_paramter_regression(nn.Module):

    # less parameters compare to LSTM
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,num_layers=num_layers,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)


    def forward(self, x):
        # run when we call this model
        #print("forward")
        # Initialize

        # h0 shape：(num_layers * num_directions, batch, hidden_size)
        # c0 shape：(num_layers * num_directions, batch, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(
            x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(
            x.device)

        # Forward propagate LSTM
        output, _ = self.lstm(x, (h0, c0))

        # take the output of the last time step, feed it in to a fully connected layer
        output = self.fc(output[-1, :, :])


        return output

    def init_network(self):
        # initialize weight and bias
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)


def training(model, x, y, batch_size):
    """

    :param model:
    :param x: shape: number of sequences,time step,  inputsize(3) (the first dimension should be the same as y
    :param y: label, shape: (num_sequences, output_size) output size is 1 for binary classification
    :param batch_size:
    :param input_size:
    :param output_size:
    :return:
    """
    # set random seed
    np.random.seed(123)
    random.seed(123)
    torch.manual_seed(123)

    num_sequences = x.shape[0]
    seq_length = x.shape[1]
    # Define training parameters
    learning_rate = 0.05 #
    num_epochs = 200

    # Convert input and target data to PyTorch datasets
    dataset = TensorDataset(x, y)

    # Create data loader for iterating over dataset in batches during training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # initialize model
    model.init_network()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, (inputs, targets) in enumerate(dataloader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            inputs = torch.permute(inputs, (
            1, 0, 2))  # change to (seq_length, batch_size,input_size)

            # Forward pass
            outputs = model(inputs)
            # outputs = outputs.float()
            targets = targets.float()
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item() * inputs.size(0)

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs,
                                                       running_loss/(num_sequences)))
    else:
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            x = torch.permute(x, (
                1, 0, 2))
            outputs = model(x.float())
            print(outputs,"r")
            print(y,"g")
            plt.plot(outputs)
            plt.plot(y)
            plt.title("r true and prediction(lr={},epoch={})".format(learning_rate,num_epochs))
            plt.savefig("lstm_result_for_logistic_model_prediction/lstm_result_training_1.png")
            plt.show()
    return model
def test_model(x,y,model):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        x = torch.permute(x, (
            1, 0, 2))
        outputs = model(x.float())
        print(outputs )
        print(y )
        plt.plot(outputs,"r")
        plt.plot(y,"g")
        plt.savefig(
            "lstm_result_for_logistic_model_prediction/lstm_result_test_1.png")
        plt.show()