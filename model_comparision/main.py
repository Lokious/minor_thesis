import torch

import lstm_regression
import reformat
import sys
import os
import pandas as pd
from torch import nn
#sys.path.append("../New folder/script/")

def plot_train_output(out_size:int, x, x_scaler, y_train, predict_outputs,
                      y_scaler, ratio_of_label, file_name:str,
                      directory_name="train"):

    # transform to original value, and set index
    predict_outputs = y_scaler.inverse_transform(predict_outputs)
    end_index = out_size + x.shape[1]
    start_index = end_index - int((end_index+1) * ratio_of_label)
    print(x.shape[1])
    print(start_index)
    print(end_index)
    predict_df = pd.DataFrame(predict_outputs.T,
                              index=range(start_index, end_index))
    y_orignal = y_scaler.inverse_transform(y_train)
    x = torch.squeeze(x)
    x = torch.permute(x, (1, 0))
    x_orignal = x_scaler.inverse_transform(x)
    x_orignal_df = pd.DataFrame(x_orignal)
    y_original_df = pd.DataFrame(y_orignal.T)

    # concat x and y for plotting
    full_time_series_df = pd.concat([x_orignal_df, y_original_df])
    full_time_series_df = full_time_series_df.reset_index(drop=True)

    full_time_series_df.to_csv("testtttttt.csv")
    import matplotlib.pyplot as plt
    #check figure save directory exit otherwise create it.
    isExist = os.path.exists("lstm_result_for_logistic_model_prediction/result_fig/{}/".format(directory_name))
    if not isExist:
        print("directory creating....")
        os.makedirs("lstm_result_for_logistic_model_prediction/result_fig/{}/".format(directory_name))
    print("save figures....")
    for i, col in enumerate(predict_df.columns):
        plt.plot(predict_df[col], "-r", label="predict")
        plt.plot(full_time_series_df[col], "-g", label="true")
        plt.xlabel("days")
        plt.ylabel("biomass")
        plt.legend(loc="upper left")
        plt.savefig(
            "lstm_result_for_logistic_model_prediction/result_fig/{}/{}_{}.png".format(directory_name,
                col,file_name))
        plt.clf()
        # plt.show()


def run_lstm_forcaseting(ratio_of_label=0.3,
    learning_rate=0.6,
    num_epoch=500,
    batch_size=10,hidden_size=2,optimize="SGD",x_name="simulated_X_data_3.csv"):
    # load data for lstm regression

    #input file x
    x = x_name
    x_train, x_test, y_train, y_test, y_train_scaler, x_train_scaler,\
    y_test_scaler, x_test_scaler,y_train_unscale = reformat.prepare_train_test_for_lstm_forecasting_multiple_dimension_label(x = x,
        label_size=ratio_of_label)
    # x_train.shape (n_seq,seq_length,n_features)

    # define model
    out_size = y_train.shape[1]
    print(out_size)
    model = lstm_regression.LSTM_forcasting(input_size=1, hidden_size=hidden_size,
                                            dropout=0, output_size=out_size)
    # model training
    print(model)
    lstm_regression.count_parameters(model)
    model, train_outputs = lstm_regression.training(model, x=x_train,
                                                    y=y_train,
                                                    batch_size=batch_size, lr=learning_rate, epoch=num_epoch,
                                                    out_fig="lstm_result_for_logistic_model_prediction/lstm_forecast_result_training_3.png")
    print(train_outputs)

    #check if we use corresponding scaler
    reformat.check_scaler(scaler=y_train_scaler,scaled_data=y_train,original_data=y_train_unscale)

    plot_train_output(out_size, x_train, x_train_scaler, y_train,
                      train_outputs, y_train_scaler, ratio_of_label,
                      file_name=x, directory_name="train_learning_rate"+
                        str(learning_rate).replace(".","_")+"epoch{}".format(num_epoch)+
                                                 "batchsize{}".format(batch_size)
                                                 +"optimizer{}".format(optimize)
                      )
    # # test model
    test_output = lstm_regression.test_model(x=x_test, y=y_test, model=model)
    criterion = nn.MSELoss(reduction='mean')
    trainloss =criterion(train_outputs, y_train).item()
    print("{}".format(trainloss))
    testloss =criterion(test_output, y_test).item()
    print("{}".format(testloss))
    print("y test: {}".format(y_test.shape))
    plot_train_output(out_size, x_test, x_test_scaler, y_test,
                      test_output, y_test_scaler, ratio_of_label,file_name=x,directory_name="test_learning_rate"+
                                                str(learning_rate).replace(".","_")+"epoch{}".format(num_epoch)+
                                                 "batchsize{}".format(batch_size)+
                                                 "optimizer{}".format(optimize))
    return trainloss, testloss

def run_lstm_parameters_prediction():
    # load data for lstm regression
    x_train, x_test, y_train, y_test, y_train_scaler, x_train_scaler = reformat.prepatre_train_test_for_lstm_regression(x="simulated_X_data_1.csv",y="simulated_Y_data_1.csv")
    # define model
    out_size = y_train.shape[1]
    print(out_size)
    model = lstm_regression.LSTM_forcasting(input_size=1, hidden_size=10,
                                            dropout=0.1, output_size=out_size)
    # model training
    print(model)
    model, predict_outputs, y = lstm_regression.training(model, x=x_train,
                                                         y=y_train,
                                                         batch_size=10,
                                                         out_fig="lstm_result_for_logistic_model_prediction/lstm_forecast_result_training_1.png")
    #print(predict_outputs)

def main():
    x = 'simulated_X_data_3'
    cross_validation_result = pd.DataFrame()
    for optimizer in ["SGD","Admin"]:
        for lr in [0.001,0.01,0.1,0.5]:
            for epoch in [10]:
                for hidden_size in [2,6,10]:
                    for ratio in [0.2,0.3,0.4,0.5]:
                        for batch_size in [10,32,64]:
                            print("optimizer:{}".format(optimizer))
                            print("lr: {}".format(lr))
                            print("hidden_size:{}".format(hidden_size))
                            print("y_ratio:{}".format(ratio))
                            print("batch_size:{}".format(batch_size))
                            tain_loss,test_loss =run_lstm_forcaseting(ratio_of_label=ratio,learning_rate=lr,
                             num_epoch=epoch, batch_size=10, hidden_size=hidden_size,
                             optimize=optimizer,x_name="{}.csv".format(x))
                            new_row = pd.DataFrame(data={"optimizer":optimizer,"lr":lr,"num_epoch":epoch,"batch_size":batch_size,"hidden_size":hidden_size,"y_ratio":ratio,"train_MSEloss":tain_loss,"test_MSEloss":test_loss},index=[0])
                            cross_validation_result = pd.concat([cross_validation_result,new_row])
                            print(cross_validation_result)
                            cross_validation_result.to_csv("{}_cross_validation_result.csv".format(x))

if __name__ == '__main__':
    main()