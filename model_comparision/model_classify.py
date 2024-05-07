import dill
import pandas as pd
import torch
import torchmetrics
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
from math import sqrt
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_percentage_error, r2_score, mean_absolute_error, accuracy_score, ConfusionMatrixDisplay, multilabel_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score, accuracy_score, plot_roc_curve, RocCurveDisplay,ConfusionMatrixDisplay, roc_auc_score, roc_curve
from torchmetrics import CohenKappa, AUROC


class LSTMModelClassification(nn.Module):

    # less parameters compare to LSTM
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,num_layers=num_layers,
                            dropout=dropout)

        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1) #https://stackoverflow.com/questions/42081257/why-binary-crossentropy-and-categorical-crossentropy-give-different-performances/46038271#46038271

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
        output_label = self.softmax(output)

        return output_label

    def init_network(self):
        # initialize weight and bias
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)


class CNNModelClassification(nn.Module):

    # treat growth curve and derivative against biomass as image input
    def __init__(self, input_size,input_channel, kernel_size, stride,ouput_size=3,
                 dropout=0.1):
        #https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        super().__init__()
        self.height = input_size[0]
        self.width = input_size[1]
        self.kernel_size = kernel_size

        #(N,C,H,W)
        #N is a batch size, C denotes a number of channels, H is a height of input planes in pixels, and W is width in pixels.
        self.cnn1 = nn.Conv2d(in_channels=input_channel,out_channels=3,kernel_size=kernel_size)
        self.Leakyrelu1 = nn.LeakyReLU()
        self.out_size1 = int(
            ((self.height - 1 * (kernel_size - 1) - 1) // 1) + 1) #output height for cnn layer
        print(self.out_size1)
        # self.cnn2 = nn.Conv2d(in_channels=6,out_channels=12,kernel_size=kernel_size)
        # self.Leakyrelu2 = nn.LeakyReLU()
        # self.out_size = int(
        #     ((self.out_size1 - 1 * (kernel_size - 1) - 1) // 1) + 1) #output height for cnn layer
        #print(self.out_size)


        pooling_kernel_size = kernel_size*3
        self.pooling1 = nn.MaxPool2d(kernel_size=pooling_kernel_size,stride=stride) #https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
        pooling_out_size = 1+(self.out_size1 - 1*(pooling_kernel_size-1)-1)//stride
        self.Leakyrelu3 = nn.LeakyReLU()
        self.pooling2 = nn.MaxPool2d(kernel_size=pooling_kernel_size,
                                     stride=stride)  # https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
        self.Leakyrelu4 = nn.LeakyReLU()
        pooling_out_size = 1 + (pooling_out_size - 1 * (pooling_kernel_size - 1) - 1) // stride
        self.cnn_1 = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=1)
        # Flatten for dense layers
        self.flatten = nn.Flatten()
        # dropout
        self.drop_out = nn.Dropout(p=dropout)
        self.linear = nn.Linear(3*pooling_out_size*pooling_out_size, ouput_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Forward propagate CNN

        cnn_output1 = self.cnn1(x)
        cnn_output1 = self.Leakyrelu3(cnn_output1)

        #print("{}".format(cnn_output2.shape))
        pooling_output1 = self.pooling1(cnn_output1)
        #print("pooling output {}".format(pooling_output1.shape))
        pooling_output2 = self.Leakyrelu3(pooling_output1)
        pooling_output2 = self.pooling2(pooling_output2)
        pooling_output = self.Leakyrelu4(pooling_output2)

        #flatten_out = self.drop_out(flatten_out)
        output = self.cnn_1(pooling_output)

        flatten_out = self.flatten(output)

        flatten_out = self.linear(flatten_out)

        output_label = self.softmax(flatten_out)
        return output_label

    def init_network(self):
        # initialize weight and bias
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

class LSTM_snp_classification(nn.Module):

    # less parameters compare to LSTM
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,num_layers=num_layers,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_size, 3)
        # output from 0 to 1
        self.softmax = nn.Softmax(dim=1)

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
        output = self.fc(output[-1, :, :],)
        output = self.softmax(output)

        return output

    def init_network(self):
        # initialize weight and bias
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)


class RandomForest():

    def RF_model(self,X_train, y_train,X_test, y_test,file_name):

        hyperparameters = {'n_estimators': [100,500],
                           'max_features': [0.3,0.5,0.7],
                           'max_depth' : [20,]
                           }

        rf_cv = GridSearchCV(
            RandomForestClassifier(random_state=0, class_weight="balanced", ),
            hyperparameters, scoring='roc_auc_ovr',
            cv=2,
            verbose=3,
            n_jobs=2)

        rf_cv.fit(X_train, y_train)
        #print(rf_cv.best_params_)
        # roc for train data
        print(X_train.shape)
        print(y_train.shape)
        self.show_roc(rf_cv.best_estimator_, X_train, y_train,
                                        (file_name + "train"))
        # roc for test data
        self.show_roc(rf_cv.best_estimator_, X_test, y_test,
                                       (file_name + "test"))
        self.plot_feature_importance(rf_cv.best_estimator_, X_train)
        # confusion matrix for train
        self.cm_display(X_train, y_train, rf_cv.best_estimator_,
                        (file_name + "train"))
        # self.cm_threshold(0.5, X_train, y_train, rf_cv.best_estimator_,
        #              (file_name + "train"))
        self.cm_display(X_test, y_test, rf_cv.best_estimator_,
                        (file_name + "test"))


        print(rf_cv.cv_results_)
        sns.lineplot(y=rf_cv.cv_results_["mean_test_score"],
                     x=rf_cv.cv_results_['param_max_depth'].data,
                     hue=rf_cv.cv_results_['param_n_estimators'])
        plt.xlabel("max_depth")
        plt.ylabel("roc_auc (mean 5-fold CV)")
        plt.title(
            "roc_auc score with different max_depth and features for RF model")
        plt.show()
        plt.close()

    def cm_display(self, x, y, rf, file_name):
        """
        The function is to plot confusion matrix with set threshold
        """
        # save drfault parameters
        from sklearn.metrics import accuracy_score
        y_predict = rf.predict(x)

        cm = confusion_matrix(y, y_predict)
        cm_display = ConfusionMatrixDisplay(cm,display_labels=["logistic","irradiance","Allee","temperature"]).plot()
        print("accuracy:{}".format(accuracy_score(y , y_predict)))
        plt.title(
            "RF confusion matrix ")

        plt.show()
        plt.close()
        return cm

    def show_roc(self,rf,X,y,file_name):
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt

        # Convert one-hot encoded labels to single-column format
        y_true = y

        # Compute the probabilities for each class
        probas = np.array(rf.predict_proba(X))

        print(probas.shape)
        # Compute the ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(4):
            fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int),
                                          probas[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot the ROC curves
        plt.figure()
        lw = 2
        colors = ['blue', 'red', 'green','orange']
        model = ["logistic","irradiance","Allee",'temperature']
        for i, color in zip(range(4), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='class {},ROC curve (area = {:.2})'.format(model.pop(0),roc_auc[i]))
        plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

    def plot_feature_importance(self,rf,X):
        # Plot feature importance
        print(len(X[0]))
        fi = pd.DataFrame(data=rf.feature_importances_,
                          index=[str(x) for x in range(1,len(X[0])+1)],
                          columns=['Importance']).sort_values(by=['Importance'], ascending=False)

        print(fi.index)
        ax1 = sns.barplot(data=fi.head(20), x="Importance",
                          y=(fi.head(20)).index)
        ax1.set_title(
            "feature importance for RF model")
        plt.tight_layout()
        plt.show()

def training(model, x, y, batch_size,epoch:int,image = False,lr=0.01,optimize = "SGD",weight=None):
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

    #criterion = nn.BCELoss()
    print(y)
    if weight:
        print("weighted classes")
        print(weight)
        class_weight = torch.tensor(weight) # weighted 3 class labels, range is in (0,1)
        # the weight is as the same order as lass label, do not need one hot encoding for label
        # the first weight is related to 'label 0' ...
        criterion = nn.CrossEntropyLoss(weight=class_weight)
    else:
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
            if not image:
                inputs = torch.permute(inputs, (
                1, 0, 2))  # change to (seq_length, batch_size,input_size)

            # Forward pass
            predict_outputs = model(inputs.float())
            # predict_outputs = predict_outputs.float()
            targets = targets.float()
            # print("before calculate loss ")
            # print(predict_outputs)
            # print(targets)
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
            if not image:
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
            accuracy = accuracy_score(y,predict_label)
            print("training result")
            print("True Negatives: {}".format(TN))
            print("False Positives: {}".format(FP))
            print("False Negatives: {}".format(FN))
            print("True Positives: {}".format(TP))
            print("accuracy:{}".format(accuracy))

    return model,accuracy

def test_model(x,y,model,image=False):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        if not image:
            x = torch.permute(x, (
                1, 0, 2))
        outputs = model(x.float())


        predict_label = torch.round(outputs)
        # print("predictlabel")
        # print(predict_label)

        cohenkappa_score = calculate_cohenkappa_score_for_tensor_prediction_out_put(
            predict_label, y)
        predict_label = convert_one_hot_endoding_to_label(predict_label)
        y = convert_one_hot_endoding_to_label(y)

        avg_auc_score = calculate_AUC_score_for_tensor_prediction_out_put(outputs,
                                                          y)
        cm = confusion_matrix(y, predict_label)
        # cm_display = ConfusionMatrixDisplay(cm, display_labels=["logistic",
        #                                                         "irradiance",
        #                                                         "Allee",
        #                                                         "temperature"]).plot()
        print("test accuracy:{}".format(accuracy_score(y, predict_label)))
        plt.title(
            "LSTM test result confusion matrix ")
        # plt.show()
        # plt.close()
        print(cm)
        accuracy = accuracy_score(y,predict_label)
        print("accuracy:{}".format(accuracy))

    return accuracy,predict_label,avg_auc_score,cohenkappa_score

def convert_one_hot_endoding_to_label(y:torch.tensor):
    label_list =[]
    for y_label in y:

        for index, label in enumerate(y_label.numpy()):

            if label ==1:
                break
        label_list.append(index)
    print(label_list)
    return label_list

def calculate_AUC_score_for_tensor_prediction_out_put(predicted_probs, true_labels):

    #convert list to tensor
    true_labels = torch.tensor(true_labels)
    print(len(true_labels.unique()))
    auroc = AUROC(task="multiclass", num_classes=len(true_labels.unique()))

    avg_auc_score = auroc(predicted_probs, true_labels).item()
    return avg_auc_score

def calculate_cohenkappa_score_for_tensor_prediction_out_put(predicted_label,true_labels_onehot):
    """ Calculate CohenKappa_score"""

    cohenkappa = CohenKappa(task="multiclass", num_classes=true_labels_onehot.shape[1])
    cohenkappa_score =cohenkappa(predicted_label, true_labels_onehot).item()
    return cohenkappa_score

def calculate_shap_importance(model, X,y):
    """
    Compute PFI (Permutation Feature Importance) for LSTM model.

    Args:
        model (torch.nn.Module): trained LSTM model.
        X (torch.Tensor): input tensor of shape (seq_len, batch_size, input_size).
        y (torch.Tensor): target tensor of shape (batch_size, output_size).

    Returns:
        importance (torch.Tensor): importance tensor of shape (seq_len, input_size).
    """
    # Set model to evaluation mode
    model.eval()

    # Initialize importance tensor
    seq_len, batch_size, input_size = X.shape
    importance = torch.zeros((seq_len, input_size))

    # Compute baseline score
    with torch.no_grad():
        y_pred = model(X)
        baseline_score = torch.nn.functional.mse_loss(y_pred, y).item()

    # Compute importance at each time step
    for i in range(seq_len):
        # Permute inputs at time step i
        X_perm = X.clone()
        X_perm[i, :, :] = torch.rand(batch_size, input_size)

        # Compute score with permuted input
        with torch.no_grad():
            y_pred_perm = model(X_perm)
            permuted_score = torch.nn.functional.mse_loss(y_pred_perm, y).item()

        # Compute importance as the decrease in score
        importance[i, :] = baseline_score - permuted_score

    # Normalize importance by dividing by the sum of all importance values
    importance = importance / torch.sum(importance)

    return importance
    return importance


def plot_top_features(feature_importance, feature_names, num_features=10):
    """
    Plots the top features based on their importance values.

    Parameters:
    -----------
    feature_importance: numpy array
        Array of feature importance values.
    feature_names: list
        List of feature names.
    num_features: int
        Number of top features to plot.

    Returns:
    --------
    None
    """

    # Get the top features based on their importance values
    top_indices = feature_importance.argsort()[::-1][:num_features]
    top_importance = feature_importance[top_indices]
    top_names = [feature_names[i] for i in top_indices]

    # Create a bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(top_names, top_importance)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.ylabel('Feature importance', fontsize=12)
    plt.title('Top {} features'.format(num_features), fontsize=14)
    plt.show()

def main():
    print(0)
if __name__ == '__main__' :
    main()