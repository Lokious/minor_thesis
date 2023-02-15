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
        hidden_size_2 = 2
        self.gru1 = nn.GRU(input_size, hidden_size, num_layers,
                             batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size_2, num_layers,
                             batch_first=True)

        self.de_gru1 = nn.GRU(hidden_size_2, hidden_size, num_layers,
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


def model_training(model,num_epochs,train_loader,no_noise_dataset,full_dataset,lr=0.0001,batch_size = 10):

    # Define the loss function and optimizer
    criterion = l2_loss2()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Training loop
    print("#######start training######")
    print("# learning rate: {}".format(lr))
    print("# batch size: {}".format(batch_size))
    for epoch in range(num_epochs):
        for data in train_loader:
            inputs = data
            optimizer.zero_grad()
            outputs = model(inputs)
            mask = mask_matrix(inputs)
            # Given an input and a target, compute a gradient according to loss function.
            loss = criterion(outputs, inputs,mask=mask)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs,
                                                       loss.item()))

    # Test the autoencoder model on the test set
    with torch.no_grad():

        test_outputs = model(full_dataset)
        print("test lost compare to input data")
        test_loss_1 = criterion(test_outputs, full_dataset)
        print('Test Loss: {:.4f}'.format(test_loss_1.item()))

        full_dataset_output = model(full_dataset)
        print("test lost compare to no_noise data")
        test_loss_2 = criterion(full_dataset_output, no_noise_dataset)
        print('Test Loss: {:.4f}'.format(test_loss_2.item()))
    return test_outputs

def data_prepare(simulated_dataset):
    """
    logistic early growth traits data
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

    return dataset,seq_len,n_features


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

def mask_matrix(input_data_with_missing_value:torch.tensor)->torch.tensor:
    """Return a tensor represent the missing value in input tensor"""

    tenosr_na = torch.isnan(input_data_with_missing_value)
    mask_tensor = tenosr_na.logical_not()

    return mask_tensor

def count_parameters(model):
    print(model)
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

class l2_loss2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,outputs, targets, mask):

        n = mask.sum()
        diff = outputs - targets
        masked_diff = mask * diff
        masked_diff_sum = masked_diff.sum()
        masked_diff_sum_squared = masked_diff_sum ** 2
        masked_sum = mask.sum()
        return masked_diff_sum_squared / masked_sum / n

def main():

    num_plants = 1200 #genotypes
    num_days = 41 #time steps
    num_features = 3 #traits
    #simulated dataset , save scaler to transform predicted value back
    '''
    ## generate simulated  dataset
    simulated_dataset = return_simulated_dataset()
    X_tensor = data_prepare(simulated_dataset)
    '''
    # use fillna LA_df
    df_LA = pd.read_csv("../data/df_LA.csv",header=0,index_col=0)
    df_height = pd.read_csv("../data/df_Height.csv",header=0,index_col=0)
    df_biomass = pd.read_csv("../data/df_Biomass.csv",header=0,index_col=0)
    dfs = [df_LA,df_height,df_biomass]
    print(dfs)
    X_tensor,num_days,num_features = data_prepare(dfs)
    print(X_tensor,num_days,num_features)
    # Shuffle the dataset
    indices = torch.randperm(num_plants)
    X_tensor = X_tensor[indices]

    # normalize whole dataset
    X_full_tensor = copy.deepcopy(X_tensor)
    X_full_dataset, _ = normalization(X_full_tensor)

    # Split the dataset into training and test sets
    train_ratio = 0.8
    train_size = int(train_ratio * num_plants)
    train_dataset = X_tensor[:train_size, :, :]
    test_dataset = X_tensor[train_size:, :, :]

    train_dataset,scaler_train = normalization(train_dataset)
    test_dataset, scaler_test = normalization(test_dataset)

    # # read no noise dataset:
    # no_noise_dfs = []
    # for trait in ["trait_1","trait_2","trait_3"]:
    #     df = pd.read_csv("../data/simulated_data/{}_without_noise.csv"
    #                      .format(trait),header=0,index_col=0)
    #     no_noise_dfs.append(df)
    # no_noise_tensor = data_prepare(no_noise_dfs)
    # no_noise_datasets,no_noise_scaler = normalization(no_noise_tensor)
    train_mask = mask_matrix(train_dataset)
    for lr in [0.1,0.01,0.001,0.0001]:
        for batch_size in [1,10]:
            input_size = num_features
            hidden_size = 12
            num_epochs = 500
            num_layers = 2
            # # Set the batch size
            # batch_size = 10

            # Create the dataloader for the training dataset
            train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                           batch_size=batch_size,
                                                           shuffle=True)


            # Create the LSTM autoencoder model
            model = LSTMAutoencoder(input_size, hidden_size, num_layers,dropout=0)
            # print parameteres
            count_parameters(model)
            # print(X_full_dataset.shape)
            # print(no_noise_datasets.shape)
            # print(test_dataset.shape)
            test_prediction = model_training(model, num_epochs, train_dataloader, test_dataset,X_full_dataset,X_full_dataset,batch_size=batch_size,lr=lr)
            # print("test prediction")
    # print(test_prediction.shape)
    test_prediction = torch.permute(test_prediction, (2, 0, 1))
    # print(test_prediction.shape)
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
