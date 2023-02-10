import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from prettytable import PrettyTable
from torch.nn import functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import time
from simulated_data import return_simulated_dataset
from sklearn.model_selection import GroupShuffleSplit
# set device: cup or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_size=2

# # Define the LSTM autoencoder model
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMAutoencoder, self).__init__()

        self.encoder = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True)
        # self.fc1 = nn.Linear(in_features=hidden_size,out_features=hidden_size)
        # self.fc2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers,
                               batch_first=True)

    def forward(self,x):
        # run when we call this model

        x, (hidden, cell) = self.encoder(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        x, (hidden, cell) = self.decoder(x)
        return x
class GRUAutoencoder(nn.Module):
    #less parameters compare to LSTM
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()

        self.gru1 = nn.GRU(input_size, hidden_size, num_layers,
                             batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size/2, num_layers,
                             batch_first=True)

        self.de_gru1 = nn.GRU(hidden_size/2, hidden_size, num_layers,
                                batch_first=True)
        self.de_gru2 = nn.GRU(hidden_size, input_size, num_layers,
                                batch_first=True)
    def forward(self,x):
        # run when we call this model

        #encoder two layer GRU
        x, hidden = self.gru1(x)
        x, hidden = self.gru2(x)
        #decoder two layer GRU
        x, hidden = self.de_gru1(x)
        x, hidden = self.de_gru2(x)
        return x
#Define the Encoder class
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(Encoder, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        x, (hidden, cell) = self.lstm(x)
        return x,hidden, cell

# Define the Decoder class
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout=0.0):
        super(Decoder, self).__init__()
        self.lstm = torch.nn.LSTM(hidden_size, output_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, hidden, cell, seq_len):
        decoder_input = torch.zeros(hidden.shape[0], seq_len, hidden_size)
        decoded, _ = self.lstm(decoder_input, (hidden, cell))
        return decoded

# Define the Autoencoder class
class Autoencoder(torch.nn.Module):
    """
    Error !!
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = Decoder(hidden_size, input_size, num_layers, dropout)

    def forward(self, x):
        x, hidden, cell = self.encoder(x)
        decoded = self.decoder(hidden, cell, x.shape[1])
        return decoded

def model_training(model,num_epochs,train_loader,test_dataset,no_noise_dataset):

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Training loop
    for epoch in range(num_epochs):
        for data in train_loader:
            inputs = data
            optimizer.zero_grad()
            outputs = model(inputs)
            # Given an input and a target, compute a gradient according to loss function.
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs,
                                                       loss.item()))

    # Test the autoencoder model on the test set
    with torch.no_grad():

        test_outputs = model(test_dataset)
        print("test lost compare to input data")
        test_loss = criterion(test_outputs, test_dataset)
        print('Test Loss: {:.4f}'.format(test_loss.item()))

        print("test lost compare to no_noise data")
        test_loss = criterion(test_outputs, no_noise_dataset)
        print('Test Loss: {:.4f}'.format(test_loss.item()))
    return test_outputs

def data_prepare(simulated_dataset):
    """
    logistic growth curve simulated data
    :return: tensor
    """
    def create_dataset(dfs):
        #(41,418,3)
        datasets = []
        for df in dfs:
            sequences = df.astype(np.float32).to_numpy().tolist()
            dataset = torch.stack([torch.tensor(s).unsqueeze(1).float() for s in sequences])
            datasets.append(dataset)

        #remove the last dimention
        tensor_dataset = torch.squeeze(torch.stack(datasets,dim=0))
        # print("shape of created dataset:")
        # print(tensor_dataset.shape)

        n_features,seq_len,n_seq= tensor_dataset.shape

        return tensor_dataset,seq_len, n_features

    dataset, seq_len, n_features = create_dataset(simulated_dataset)
    # the number represent the number in old order, and after that will place it
    # as the new order for example: permute(2,0,1)
    # (a,b,c) -> (c,a,b)
    dataset = torch.permute(dataset, (2, 1, 0))

    return dataset

def generate_dataset2():

    # Generate random simulated plant trait data
    np.random.seed(0)
    num_plants = 418 #genotypes
    num_days = 41 #time steps
    num_features = 3 #traits

    days = np.arange(num_days)
    X = np.zeros((num_plants, num_days, num_features))
    for i in range(num_plants):
        for j in range(num_features):
            X[i, :, j] = np.sin(
                2 * np.pi * days / num_days + 2 * np.pi * j / num_features) + np.random.normal(
                0, 0.1, num_days)
    # Convert the numpy array to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    return X_tensor

def normalization(dataset:torch.tensor)->torch.tensor:

    # normalization
    # train the normalization
    from sklearn.preprocessing import StandardScaler
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
    dataset = torch.permute(timeseries_tensor, (1, 2, 0))  # (time step, number of sequences, inputsize(3))
    #dataset = torch.permute(dataset, (2, 1, 0))
    # print("after normalize")
    # print(dataset.shape)
    return dataset,scaler


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def main():

    num_plants = 418 #genotypes
    num_days = 41 #time steps
    num_features = 3 #traits
    #simulated dataset , save scaler to transform predicted value back
    ## generate simulated  dataset
    simulated_dataset = return_simulated_dataset()
    X_tensor = data_prepare(simulated_dataset)
    # #simulated dataset 2
    # X_tensor = generate_dataset2()

    # Shuffle the dataset
    indices = torch.randperm(num_plants)
    X_tensor = X_tensor[indices]

    # Split the dataset into training and test sets
    train_ratio = 0.8
    train_size = int(train_ratio * num_plants)
    train_dataset = X_tensor[:train_size, :, :]
    test_dataset = X_tensor[train_size:, :, :]

    train_dataset,scaler_train = normalization(train_dataset)
    test_dataset, scaler_test = normalization(test_dataset)
    # read no noise dataset:
    no_noise_dfs = []
    for trait in ["trait_1","trait_2","trait_3"]:
        df = pd.read_csv("../data/simulated_data/{}_without_noise.csv"
                         .format(trait),header=0,index_col=0)
        no_noise_dfs.append(df)
    no_noise_tensor = data_prepare(no_noise_dfs)
    no_noise_datasets,no_noise_scaler = normalization(no_noise_tensor)
    # Set the batch size
    batch_size = 1

    # Create the dataloader for the training dataset
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)


    # Create the LSTM autoencoder model
    input_size = num_features
    hidden_size = 2
    num_epochs = 100
    num_layers = 1
    model = LSTMAutoencoder(input_size, hidden_size, num_layers,dropout=0.0)
    # print parameteres
    count_parameters(model)
    test_prediction = model_training(model, num_epochs, train_dataloader, test_dataset,no_noise_datasets)
    print("test prediction")
    print(test_prediction.shape)
    test_prediction = torch.permute(test_prediction, (2, 0, 1))
    print(test_prediction.shape)
    for i,trait in enumerate(test_prediction):
        # inverse transform and print
        # print(trait.shape)
        inversed = scaler_test.inverse_transform(trait)
        # print(inversed.shape)
        inversed = pd.DataFrame(inversed)
        inversed.to_csv("../data/simulated_data/trait_{}_predict.csv".format((i+1)))
        # print(inversed)

if __name__ == '__main__':
    main()
