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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_percentage_error, r2_score, mean_absolute_error, accuracy_score, ConfusionMatrixDisplay, multilabel_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score, accuracy_score, plot_roc_curve, RocCurveDisplay,ConfusionMatrixDisplay, roc_auc_score, roc_curve


class LSTMModelClassification(nn.Module):

    # less parameters compare to LSTM
    def __init__(self, input_size, hidden_size,ouput_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,num_layers=num_layers,
                            dropout=dropout)

        self.fc = nn.Linear(hidden_size, ouput_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # run when everytime we call this model
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
        output_label = self.sigmoid(output)

        return output_label

    def init_network(self):
        # initialize weight and bias
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)


class CNNModelClassification(nn.Module):

    # less parameters compare to LSTM
    def __init__(self, input_size, kernel_size, ouput_size,time_step, num_layers=2,
                 dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.cnn = nn.Conv1d(in_channels=time_step,out_channels=time_step*kernel_size,kernel_size=kernel_size,)
        self.fc = nn.Linear(time_step*kernel_size, ouput_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Forward propagate LSTM
        output, _ = self.cnn(x)

        # take the output of the last time step, feed it in to a fully connected layer
        output = self.fc(output[-1, :, :])
        output_label = self.sigmoid(output)

        return output_label

    def init_network(self):
        # initialize weight and bias
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)


class RandomForest():

    def RF_model(self,X_train, y_train,X_test, y_test,file_name):

        hyperparameters = {'n_estimators': [100,500,1000],
                           'max_features': [0.3,0.5,0.7],
                           'max_depth' : [10,15,20]
                           }
        rf_cv = GridSearchCV(
            RandomForestClassifier(random_state=0, class_weight="balanced", ),
            hyperparameters, scoring='roc_auc',
            cv=5,
            verbose=3,
            n_jobs=2)

        rf_cv.fit(X_train, y_train)
        print(rf_cv.best_params_)
        # roc for train data
        threshold_train = self.show_roc(rf_cv.best_estimator_, X_train, y_train,
                                        (file_name + "train"))
        # roc for test data
        threshold_test = self.show_roc(rf_cv.best_estimator_, X_test, y_test,
                                       (file_name + "test"))

        # confusion matrix for train
        self.cm_threshold(threshold_train, X_train, y_train, rf_cv.best_estimator_,
                          (file_name + "train"))
        # self.cm_threshold(0.5, X_train, y_train, rf_cv.best_estimator_,
        #              (file_name + "train"))
        self.cm_threshold(threshold_test, X_test, y_test, rf_cv.best_estimator_,
                          (file_name + "test"))
        cm_matrix = self.cm_threshold(threshold_train, X_test, y_test,
                                      rf_cv.best_estimator_,
                                      (file_name + "test (same as train threshold)"))

        print(rf_cv.cv_results_)
        sns.lineplot(y=rf_cv.cv_results_["mean_test_score"],
                     x=rf_cv.cv_results_['param_max_depth'].data,
                     hue=rf_cv.cv_results_['param_n_estimators'])
        plt.xlabel("max_depth")
        plt.ylabel("roc_auc (mean 5-fold CV)")
        plt.title(
            "roc_auc score with different max_depth and features for RF model")
        plt.close()

    def cm_threshold(self, threshold, x, y, rf, file_name):
        """
        The function is to plot confusion matrix with set threshold
        """
        # save drfault parameters
        print(plt.rcParams)
        default_para = plt.rcParams
        y_pre_threshold = []
        for point in rf.predict_proba(x):
            if point[1] >= threshold:
                y_pre_threshold.append(1)
            else:
                y_pre_threshold.append(0)
        else:

            cm = confusion_matrix(y, y_pre_threshold)
            fig, ax = plt.subplots()

            cm_display = ConfusionMatrixDisplay(cm).plot()

            plt.title(
                "RF confusion matrix threshold:{}".format(
                    threshold))
            cm_display.figure_.savefig(
                '../autodata/separate_seed_result/cm_threshold{}_{}.svg'.format(
                    threshold, file_name), dpi=300)
            plt.show()
            plt.close()
        return cm

    def show_roc(self,rf,X,y,file_name):
        from sklearn import metrics

        # #save drfault parameters
        # roc for training data
        y_probs = rf.predict_proba(X)
        # keep probabilities for the positive outcome only
        yhat = y_probs[:, 1]
        print(yhat)
        fpr, tpr, thresholds = metrics.roc_curve(y, yhat)
        roc_auc = metrics.auc(fpr, tpr)
        gmeans = []
        for point in range(len(fpr)):
            gmean = sqrt(tpr[point] * (1 - fpr[point]))
            gmeans.append(gmean)
        # # locate the index of the largest g-mean
        ix = np.argmax(gmeans)
        print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr,
                                          roc_auc=roc_auc).plot()
        plt.title('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
        display.figure_.savefig("../autodata/separate_seed_result/RF_ROC_curve_{}_data.svg".format(file_name))
        #plt.show()
        plt.close()
        ###precision_recall###
        from sklearn.metrics import PrecisionRecallDisplay

        display = PrecisionRecallDisplay.from_estimator(
            rf, X, y, name="Random Forest"
        )
        precision_recall_figure = display.ax_.set_title("Precision-recall curve")

        plt.show()
        plt.close()
        return thresholds[ix]


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

    criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    if optimize == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimize == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)


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

    else:

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            x = torch.permute(x, (
                1, 0, 2))
            predict_outputs = model(x.float())
            # print(predict_outputs)
            # print(y)
            # print("predict_y_shape {}".format(predict_outputs.shape))
            # print("true y_shape {}".format(y.shape))
            predict_label = torch.round(predict_outputs)
            # print("predictlabel")
            # print(predict_label)
            y = convert_one_hot_endoding_to_label(y)
            predict_label = convert_one_hot_endoding_to_label(predict_label)
            cm = confusion_matrix(y, predict_label)
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
            print("accuracy:{}".format(((TN+TP)/(FP+FN+TP+TN))))

    return model,((TN+TP)/(FP+FN+TP+TN))

def test_model(x,y,model):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        x = torch.permute(x, (
            1, 0, 2))
        outputs = model(x.float())
        predict_label = torch.round(outputs)
        # print("predictlabel")
        # print(predict_label)
        predict_label = convert_one_hot_endoding_to_label(predict_label)
        y = convert_one_hot_endoding_to_label(y)
        cm = confusion_matrix(y, predict_label)

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
        print("accuracy:{}".format((TN+TP)/(FP+FN+TP+TN)))

    return ((TN+TP)/(FP+FN+TP+TN))

def convert_one_hot_endoding_to_label(y:torch.tensor):
    label_list =[]
    for y_label in y:

        for index, label in enumerate(y_label.numpy()):

            if label ==1:
                break
        label_list.append(index)
    print(label_list)
    return label_list