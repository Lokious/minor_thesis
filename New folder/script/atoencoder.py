import torch
from torch import nn
import torch.optim as optim

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
hidden_size=128

# # Define the LSTM autoencoder model
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMAutoencoder, self).__init__()

        self.encoder = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(in_features=hidden_size,out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers,
                               batch_first=True)

    def forward(self,x):
        # run when we call this model

        x, (hidden, cell) = self.encoder(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x, (hidden, cell) = self.decoder(x)
        return x
#Define the Encoder class
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(Encoder, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell

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
        hidden, cell = self.encoder(x)
        decoded = self.decoder(hidden, cell, x.shape[1])
        return decoded

def model_training(model,num_epochs,train_loader,test_dataset):

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Training loop
    for epoch in range(num_epochs):
        for data in train_loader:
            inputs = data

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs,
                                                       loss.item()))

    # Test the autoencoder model on the test set
    with torch.no_grad():
        test_inputs = test_dataset
        test_outputs = model(test_inputs)
        print(test_outputs)
        test_loss = criterion(test_outputs, test_inputs)
        print('Test Loss: {:.4f}'.format(test_loss.item()))
    return test_outputs
def data_prepare():
    """
    logistic growth curve simulated data
    :return: tensor
    """
    # generate simulated  dataset
    simulated_dataset = return_simulated_dataset()
    simulated_dataset = simulated_dataset
    # normalization
    # train the normalization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    timeseries_df = []
    for feature_df in simulated_dataset:
        scaler = scaler.fit(feature_df)
        # normalize the dataset and print
        standardized = scaler.transform(feature_df)
        print(standardized.shape)
        timeseries_df.append(standardized)


    def create_dataset(dfs):
        #(41,418,3)
        datasets = []
        for df in dfs:
            #to_numpy().tolist()
            sequences = df.astype(np.float32).tolist()

            dataset = torch.stack([torch.tensor(s).unsqueeze(1).float() for s in sequences])
            datasets.append(dataset)
        #remove the last dimention
        tensor_dataset = torch.squeeze(torch.stack(datasets,dim=0))
        print(tensor_dataset.shape)

        n_features,seq_len,n_seq= tensor_dataset.shape
        ##################
        #seq_len:time step=60; n_features=3

        return tensor_dataset,seq_len, n_features

    dataset, seq_len, n_features = create_dataset(timeseries_df)

    #put time step at the first dimention
    dataset = torch.permute(dataset, (2, 1, 0))#torch.Size([418, 41, 3])
    #print(dataset.shape)
    return dataset,scaler

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

def main():

    num_plants = 418 #genotypes
    num_days = 41 #time steps
    num_features = 3 #traits
    #simulated dataset , save scaler to transform predicted value back
    X_tensor,scaler = data_prepare()
    # #simulated dataset 2
    # X_tensor = generate_dataset2()

    print(X_tensor.shape)
    # Shuffle the dataset
    indices = torch.randperm(num_plants)
    X_tensor = X_tensor[indices]

    # Split the dataset into training and test sets
    train_ratio = 0.8
    train_size = int(train_ratio * num_plants)
    train_dataset = X_tensor[:train_size, :, :]
    test_dataset = X_tensor[train_size:, :, :]

    # Set the batch size
    batch_size = 1

    # Create the dataloader for the training dataset
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True)

    # Create the dataloader for the test dataset
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)

    # Create the LSTM autoencoder model
    input_size = 3
    hidden_size = 128
    num_epochs = 100
    num_layers = 1
    model = LSTMAutoencoder(input_size, hidden_size, num_layers,dropout=0.0)

    # Set the model to the GPU
    #model = model.cuda() if torch.cuda.is_available() else model

    # #Define the loss function and optimizer
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    test_prediction = model_training(model, num_epochs, train_dataloader, test_dataset)

    # inverse transform and print
    inversed = scaler.inverse_transform(test_prediction)
    print(inversed)
    print(inversed.shape)
if __name__ == '__main__':
    main()
