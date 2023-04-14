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

    #print(dataset.shape)
    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    scaler = StandardScaler()
    timeseries_tensor = []

    dataset = torch.permute(dataset, (2, 1, 0))
    #print(dataset.shape)
    tensors = list(dataset)
    # print(tensors)
    scalers =[]
    for feature_tensor in tensors:
        # due to percision issue, there will be a slightly difference between the
        # original and scaler.inverse_transformed data, use double() to reduce the difference,
        # However, it may still happen.
        # print("size of the input of scaler")
        # print(feature_tensor.shape)
        scaler = scaler.fit(feature_tensor)
        scalers.append(copy.deepcopy(scaler)) ##save object, remember deepcopoy!
        #print("shape before scaling:{}".format(feature_tensor.shape))
        # normalize the dataset and print
        standardized = scaler.transform(feature_tensor)
        #print(standardized.shape)
        standardized = torch.stack([torch.tensor(s).double() for s in standardized])
        timeseries_tensor.append(standardized)
        #unscale to check the scaling result
        scale_inverse = scaler.inverse_transform(standardized)

        scale_inverse = torch.stack([torch.tensor(s).double() for s in scale_inverse])

        print("check inside normalization")
        print(torch.sum(torch.eq(feature_tensor, scale_inverse)).item() / feature_tensor.nelement())
    timeseries_tensor = torch.stack(timeseries_tensor)
    # print("before permute")
    # print(timeseries_tensor.shape)
    # put time step at the first dimention
    dataset = torch.permute(timeseries_tensor, (2, 1, 0))  # (time step, number of sequences, inputsize(3))

    # print("after normalize")
    #print(dataset.shape)
    return dataset,scalers

class LSTM_paramter_regression(nn.Module):

    # less parameters compare to LSTM
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.01):
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


class LSTM_forcasting(nn.Module):

    # less parameters compare to LSTM
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1,output_size=28):
        super().__init__()
        self.input_size = input_size
        self.out_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,num_layers=num_layers,
                            dropout=dropout)

        self.fc = nn.Linear(in_features=hidden_size,out_features=output_size)

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
        output_1, _ = self.lstm1(x, (h0, c0))
        # take the output of the last time step, feed it in to a fully connected layer
        output = self.fc(output_1[-1, :, :])
        return output

    def init_network(self):
        # initialize weight and bias
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

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


def training(model, x, y, batch_size,epoch:int,out_fig="lstm_result_for_logistic_model_prediction/lstm_result_training_3.png",lr=0.01,optimize = "SGD"):
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
    print(num_sequences)
    seq_length = x.shape[1]
    # Define training parameters
    learning_rate = lr #
    num_epochs = epoch

    # Convert input and target data to PyTorch datasets
    dataset = TensorDataset(x, y)

    # Create data loader for iterating over dataset in batches during training
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # initialize model
    model.init_network()

    # Define loss function and optimizer

    criterion = nn.MSELoss(reduction = 'mean')
    #criterion = nn.CrossEntropyLoss()
    if optimize == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimize == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_list = []
    x_axis = []
    # plt.ion()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # Train the model
    for epoch in range(num_epochs):
        running_loss = 0.0 #running loss for every epoch

        for i, (inputs, targets) in enumerate(dataloader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            inputs = torch.permute(inputs, (
            1, 0, 2))  # change to (seq_length, batch_size,input_size)

            # Forward pass
            predict_outputs = model(inputs.float())

            # predict_outputs = predict_outputs.float()
            targets = targets.float()
            loss = criterion(predict_outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # sum the running loss for every batch, the last batch maybe smaller
            running_loss += loss.item()*inputs.shape[1] # loss.item() is the mean loss of the total batch

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs,
                                                       running_loss/(num_sequences)))
            # loss_list.append(running_loss/(num_sequences))
            # x_axis.append(epoch + 1)
            # line1, = ax.plot(x_axis, loss_list, 'b-')
            # ax.set_ylabel("loss")
            # line1.set_ydata(loss_list)
            # line1.set_xdata(x_axis)
            # fig.canvas.draw_idle()
            # fig.canvas.flush_events()
            # plt.show(block=False)
    else:
        # print(x_axis,loss_list)
        # plt.plot(x_axis, loss_list, 'b-')
        # plt.ylabel("loss")
        # plt.xlabel("epoch")
        # plt.savefig("lr{}_epoch{}_batch_size{}_optim{}.png".format(lr,epoch,batch_size,optimize))
        # plt.clf()
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            x = torch.permute(x, (
                1, 0, 2))
            predict_outputs = model(x.float())
            print(predict_outputs)
            print(y)
            print("predict_y_shape {}".format(predict_outputs.shape))
            print("true y_shape {}".format(y.shape))
            plot_r_predict_against_true(predict_outputs, y,"train")

    return model,predict_outputs

def test_model(x,y,model,out_fig="lstm_result_for_logistic_model_prediction/lstm_result_test_3.png"):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        x = torch.permute(x, (
            1, 0, 2))
        outputs = model(x.float())
        print("predict_y_shape {}".format(outputs.shape))
        print("true y_shape {}".format(y.shape))
        plot_r_predict_against_true(outputs, y,"test")
    return outputs


def plot_r_predict_against_true(outputs,y,title="train"):
    """
    scatter plot for r true and predict, use a line to connect true and predict value
    :param outputs:
    :param y:
    :param title:
    :return:
    """

    outputs = outputs.flatten()
    predict_r = outputs.tolist()
    y = y.flatten()
    true_r = y.tolist()
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(predict_r,true_r)
    print("{}MSE".format(title,mse))
    x = list(range(len(predict_r)))
    from matplotlib import collections as matcoll
    y_tuple = list(zip(predict_r, true_r))

    lines = []
    for i, j in zip(x, y_tuple):
        pair = [(i, j[0]), (i, j[1])]
        lines.append(pair)

    linecoll = matcoll.LineCollection(lines, colors='k')
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots()

    ax.plot(x, [i for (i, j) in y_tuple], 'rs', markersize=4,label="predict")
    ax.plot(x, [j for (i, j) in y_tuple], 'bo', markersize=4,label="true")
    ax.add_collection(linecoll)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot((x, x), ([i for (i, j) in y_tuple], [j for (i, j) in y_tuple]), c='black')
    plt.ylabel("r")
    plt.legend(loc='best')
    plt.xlabel("samples")
    plt.title("{} result MSE:{:.2}".format(title,mse))
    # plt.xticks(list(range(y.shape[0])))
    plt.show()

def main():
    model = LSTM_forcasting(input_size=1,hidden_size=2,output_size=120)
    input_deri = \
        pd.read_csv("data/simulated_data/fixed_Max_range_of_parameters/simulated_derivative_data_irradiance_time_independent_noise_0.25.csv",header=0,index_col=0)
    input_deri = torch.tensor(input_deri.values)
    input_deri = torch.unsqueeze(input_deri,1)
    input_deri = torch.permute(input_deri,(2,0,1))
    print(input_deri.shape)
    output_y = copy.deepcopy(input_deri)
    output_y = torch.squeeze(output_y)
    training(model=model,x=input_deri,y=output_y,batch_size=10,epoch=100,lr=0.001)
if __name__ == '__main__':
    main()