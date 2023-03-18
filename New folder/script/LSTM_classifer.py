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


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.0):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = 1

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True,dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(
            x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(
            x.device)

        out, _ = self.lstm(x, (h0, c0))
        output = self.fc(out[-1, :, :])
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
    learning_rate = 0.0001 #0.001 too large? when batch 100 it gives better result than batch 10 or 1
    num_epochs = 100

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

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs,
                                                       running_loss/(num_sequences)))

    else:
        model.eval()  # Set the model to evaluation mode

        with torch.no_grad():
            x = torch.permute(x, (
                1, 0, 2))
            outputs = model(x.float())

            predict_label = torch.round(outputs)

            cm = confusion_matrix(y,predict_label)
            try:
                print(cm)
                TN = cm[0, 0]
                FP = cm[0, 1]
                FN = cm[1, 0]
                TP = cm[1, 1]
                print("training result")
                print("True Negatives: {}".format(TN))
                print("False Positives: {}".format(FP))
                print("False Negatives: {}".format(FN))
                print("True Positives: {}".format(TP))

                file = open("output_lstm.txt", "a")
                file.write("training loss at last epoch: {}\n".format(
                    (running_loss / num_sequences)))
                file.write("Training result:")
                file.write("True Negatives: {}\n".format(TN))
                file.write("False Positives: {}\n".format(FP))
                file.write("False Negatives: {}\n".format(FN))
                file.write("True Positives: {}\n".format(TP))
                file.write("test loss: {}\n".format(loss))
            except:
                file = open("output_lstm.txt", "a")
                file.write("training loss at last epoch: {}\n".format(
                    (running_loss / num_sequences)))
                file.write("error")
                file.write(" loss: {}\n".format(loss))



def model_evaluation(model,x,y,snp,cross_validation=False):
    if not cross_validation:
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            x = torch.permute(x, (
                1, 0, 2)).float()  # change to (seq_length, batch_size,input_size)
            # Define loss function and optimizer
            # https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
            criterion = nn.BCELoss()
            outputs = model(x)

            loss = criterion(outputs.float(), y.float())

            print("test loss: {}\n".format(loss))
            file = open("output_lstm.txt", "a")
            file.write("--------------------------\n")
            file.write("test loss: {}\n".format(loss))
            try:

                #save confusion matrix
                predict_label = torch.round(outputs)
                cm = confusion_matrix(y,predict_label)
                print(cm)
                TN = cm[0, 0]
                FP = cm[0, 1]
                FN = cm[1, 0]
                TP = cm[1, 1]

                print("True Negatives: {}".format(TN))
                print("False Positives: {}".format(FP))
                print("False Negatives: {}".format(FN))
                print("True Positives: {}".format(TP))

                file = open("output_lstm.txt", "a")
                file.write("True Negatives: {}\n".format(TN))
                file.write("False Positives: {}\n".format(FP))
                file.write("False Negatives: {}\n".format(FN))
                file.write("True Positives: {}\n".format(TP))
                file.write("test loss: {}\n".format(loss))

                classes = ['Negative', 'Positive']
                plt.plot()
                # Create the confusion matrix plot using seaborn
                sns.heatmap(cm, annot=True, fmt='g', xticklabels=classes, yticklabels=classes)

                # Set the axis labels and title
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.title('Confusion Matrix')

                # Save the plot as a PNG file
                plt.savefig('{}_bilstm.png'.format(snp),dpi=300)
                plt.clf()
                matthews_corrcoef = BinaryMatthewsCorrCoef()
                mcc = matthews_corrcoef(predict_label, y)
                print("mcc {}".format(mcc))
                return loss,TN,FP,TP,FN,mcc
            except:
                file = open("output_lstm.txt", "a")
                file.write("error")
                file.write("test loss: {}\n".format(loss))
                return loss, "NA", "NA", "NA", "NA","NA"
    else:
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            x = torch.permute(x, (
                1, 0,
                2)).float()  # change to (seq_length, batch_size,input_size)
            # Define loss function and optimizer
            # https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
            criterion = nn.BCELoss()
            outputs = model(x)

            loss = criterion(outputs.float(), y.float())
            predict_label = torch.round(outputs)
            print("test loss: {}\n".format(loss))
            matthews_corrcoef = BinaryMatthewsCorrCoef()
            mcc = matthews_corrcoef(predict_label, y)
            print("mcc: {}".format(mcc))
            try:

                cm = confusion_matrix(y, predict_label)
                print(cm)
                TN = cm[0, 0]
                FP = cm[0, 1]
                FN = cm[1, 0]
                TP = cm[1, 1]
                return loss, TN, FP, TP, FN, mcc
            except:
                return loss, "NA", "NA", "NA", "NA","NA"

def coss_validation_for_best_model(snp_list,input_x,hidden_size_range=[2,1,3],dropouts=[0.01,0.1],batch_sizes=[10,100]):


    # Some authors suggest that when multiplying batch size by k,
    # we should also multiply the learning rate with \sqrt{k} to keep the variance in the gradient expectation constant
    test_result = {}
    print("########corss validation with  hideensize {} dropout {}".format(hidden_size_range,dropouts))
    print("# Total {} fold, for {} SNPs #".format(len(hidden_size_range)*len(dropouts),len(snp_list)))

    for i,snp in enumerate(snp_list):
        print("####### SNP {}: {} ######".format(i,snp))
        # x:(time step, number of sequences, inputsize(3))
        # y: (number of sequences, class_number)

        try:
            input_y = dill.load(open(
                "../data/input_data/spline_predict_input_average/input_Y_{}".format(
                    snp), "rb"))
        except:
            continue
        # print(input_x.shape)
        # print(input_y.shape)
        test_result[snp]={'mcc':-1}
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(input_x, input_y,
                                                            test_size=0.3,
                                                            random_state=1)
        # try different combination of hidden size dropout and batch size
        for hidden_size in hidden_size_range:
            for dropout in dropouts:
                for batch_size in batch_sizes:
                    print("###hidden_size:{} dropout:{} batch_size:{}###".format(hidden_size,dropout,batch_size))
                    # define model
                    model = LSTM_classification(input_size=6, hidden_size=hidden_size, dropout=dropout)
                    # model training
                    print(model)

                    training(model, x=x_train, y=y_train, batch_size=batch_size)

                    # model evaluation
                    #print(y_test)
                    loss,TN,FP,TP,FN,mcc = model_evaluation(model,x_test,y_test,snp)
                    if mcc!= "NA" and (mcc > test_result[snp]["mcc"]):
                        test_result[snp]["mcc"] = mcc
                        test_result[snp]["hidden_size"] = hidden_size
                        test_result[snp]["dropout"] = dropout
                        test_result[snp]["batch_size"] = batch_size
        else:
            cross_validation_df = pd.DataFrame(test_result)
            cross_validation_df.to_csv("crossvalidation.csv")
            print(cross_validation_df)
    else:
        with open("../result/log_spline_predict_result_lstm_DH_LA_and_Height/coss_validation/result_dictionary","wb") as dillfile:
            dill.dump(test_result,dillfile)

def main():

    # read and pre-process input (build model for each snp)
    # snp_df = pd.read_csv("../data/chosen_snps_map.csv", header=0, index_col=0) # all chosen snps
    # snp_list = list(snp_df.index)
    input_x = dill.load(open("../data/input_data/spline_predict_input_average/input_X","rb"))
    print(input_x.shape)
    from autoencoder import normalization
    input_x,scaler = normalization(input_x)

    # # after average based on genotype, nan only at start
    # print("number of missing days")
    # print(input_x[:,:,0])
    # df1 = pd.DataFrame(input_x[:,:,3].numpy())
    # print(df1.isna().sum())

    #fill na with 0 cause there are all at start
    input_x = torch.nan_to_num(input_x)
    snp_df = pd.read_csv("../result/log_spline_predict_result_lstm_DH_LA_and_Height/result.csv", header=0, index_col=0) # subset which maybe related to traits
    snp_df = snp_df.dropna()
    print(snp_df)
    snp_list = list(snp_df["snps"])
    coss_validation_for_best_model(input_x=input_x,snp_list=snp_list)

    '''
    test_result = {}
    related_snps = []
    for snp in snp_list:

        # x:(time step, number of sequences, inputsize(3))
        # y: (number of sequences, class_number)
        with open("output_lstm.txt", "a") as f:
            f.write("#####{}#######\n".format(snp))
        try:
            input_y = dill.load(open("../data/input_data/spline_predict_input/input_Y_{}".format(snp),"rb"))
        except:
            continue
        # print(input_x.shape)
        # print(input_y.shape)
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(input_x, input_y,
                                                            test_size=0.3,
                                                            random_state=1)
        # define model
        model = LSTM_classification(input_size=6, hidden_size=2, dropout=0.01)
        # model training
        print(model)

        training(model, x=x_train, y=y_train, batch_size=10)

        # model evaluation
        #print(y_test)
        loss,TN,FP,TP,FN,mcc = model_evaluation(model,x_test,y_test,snp)
        try:
            if ((TN +FN) !=0) and ((TP + FP) != 0):
                related_snps.append(snp)
                save_df = pd.DataFrame(related_snps)
                print(save_df)
                save_df.to_csv("possiable_related_snps_log.csv")
        except:
            continue
        test_result[snp] = [loss.item(),TN,FP,TP,FN,mcc]
        # save to csv file
        test_result_df=pd.DataFrame.from_dict(test_result, orient='index',
                               columns=['test_loss','TN','FP','TP','FN',"mcc"])
        test_result_df.to_csv("LSTM_test_result_log.csv")
    '''
if __name__ == '__main__':
    main()
