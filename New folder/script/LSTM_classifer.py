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

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class LSTM_classification(nn.Module):

    # less parameters compare to LSTM
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,num_layers=num_layers,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        # output from 0 to 1
        self.sigmoid = nn.Sigmoid()

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
        output = self.sigmoid(output)

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
    num_sequences = x.shape[0]
    seq_length = x.shape[1]
    # Define training parameters
    learning_rate = 0.0001
    num_epochs = 200

    # Convert input and target data to PyTorch datasets
    dataset = TensorDataset(x, y)

    # Create data loader for iterating over dataset in batches during training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # initialize model
    model.init_network()

    # Define loss function and optimizer
    criterion = nn.BCELoss()
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

            # if (i + 1) % 10 == 0:
            #     print('Epoch [%d], batch [%d], loss: %.3f' % (
            #         epoch + 1, i + 1, running_loss / ((i + 1) * batch_size)))
        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs,
                                                       running_loss / (num_sequences)))
        # print('Epoch [%d], loss: %.3f' % (
        #     epoch + 1, running_loss / num_sequences))
    else:
        file = open("output.txt", "a")
        file.write("training loss at last epoch: {}\n".format(
            (running_loss / num_sequences)))


def model_evaluation(model,x,y,snp):

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        x = torch.permute(x, (
            1, 0, 2)).float()  # change to (seq_length, batch_size,input_size)
        # Define loss function and optimizer
        print(x)
        criterion = nn.BCELoss()
        outputs = model(x)
        print(outputs.shape,y.shape,x.shape)
        loss = criterion(outputs.float(), y.float())

        print("test loss: {}\n".format(loss))
        file = open("output.txt", "a")
        file.write("--------------------------\n")
        file.write("test loss: {}\n".format(loss))

        #save confusion matrix
        predict_label = torch.round(outputs)
        cm = confusion_matrix(y,predict_label)
        plt.imshow(cm, cmap=plt.cm.Blues)
        plt.colorbar()
        plt.xticks([0, 1])
        plt.yticks([0, 1])
        plt.xlabel('Predicted label')
        plt.ylabel('True label')

        # Add values
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j],
                         ha='center', va='center', color='white')

        # Save the plot as a PNG file
        plt.savefig('{}.png'.format(snp))
def main():

    # read and pre-process input (build model for each snp)
    snp_df = pd.read_csv("../data/chosen_snps_map.csv", header=0, index_col=0)
    snp_list = list(snp_df.index)
    input_x = dill.load(open("../data/input_data/input_X","rb"))
    for snp in snp_list:

        # x:(time step, number of sequences, inputsize(3))
        # y: (number of sequences, class_number)
        with open("output.txt", "a") as f:
            f.write("#####{}#######\n".format(snp))
        input_y = dill.load(open("../data/input_data/input_Y_{}".format(snp),"rb"))
        # print(input_x.shape)
        # print(input_y.shape)
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(input_x, input_y,
                                                            test_size=0.3,
                                                            random_state=1)
        # define model
        model = LSTM_classification(input_size=3, hidden_size=2, dropout=0.01)
        # model training
        print(model)
        training(model, x=x_train, y=y_train, batch_size=10)

        # model evaluation
        model_evaluation(model,x_test,y_test,snp)

if __name__ == '__main__':
    main()
