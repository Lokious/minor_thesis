import dill
import torch

# import from current directory
import lstm_regression
import model_classify
import reformat

import sys
import os
import pandas as pd
from torch import nn
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
                              index=range(0,start_index))

    y_orignal = y_scaler.inverse_transform(y_train)
    x = torch.squeeze(x)
    x = torch.permute(x, (1, 0))
    x_orignal = x_scaler.inverse_transform(x)
    x_orignal_df = pd.DataFrame(x_orignal)
    y_original_df = pd.DataFrame(y_orignal.T)
    x_orignal_df.T.to_csv("original.csv")
    predict_df.T.to_csv("predict.csv")
    # concat x and y for plotting
    #full_time_series_df = pd.concat([x_orignal_df, y_original_df])
    #full_time_series_df = full_time_series_df.reset_index(drop=True)
    full_time_series_df = x_orignal_df
    #full_time_series_df.to_csv("testtttttt.csv")
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
        plt.show()
        # plt.savefig(
        #     "lstm_result_for_logistic_model_prediction/result_fig/{}/{}_{}.png".format(directory_name,
        #         col,file_name))
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
    #reformat.check_scaler(scaler=y_train_scaler,scaled_data=y_train,original_data=y_train_unscale)

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
    print("test output")
    print(train_outputs)

    # plot_train_output(out_size, x_test, x_test_scaler, y_test,
    #                   test_output, y_test_scaler, ratio_of_label,file_name=x,directory_name="test_learning_rate"+
    #                                             str(learning_rate).replace(".","_")+"epoch{}".format(num_epoch)+
    #                                              "batchsize{}".format(batch_size)+
    #                                              "optimizer{}".format(optimize))
    return trainloss, testloss

def run_lstm_parameters_prediction():
    # load data for lstm regression
    x_train, x_test, y_train, y_test = reformat.prepatre_train_test_for_lstm_regression(x="simulated_X_data_1.csv",y="simulated_Y_data_1.csv")
    # define model
    out_size = y_train.shape[1]
    print(out_size)
    model = lstm_regression.LSTM_paramter_regression(input_size=1, hidden_size=3,
                                            dropout=0.1)
    # model training
    print(model)
    model,ouput_predict = lstm_regression.training(model, x=x_train,
                                             y=y_train,
                                             batch_size=10,epoch=300,lr=0.01,optimize="Adam")
    lstm_regression.test_model(x_test,y_test,model)
#print(predict_outputs)

def run_LSTM_model_for_biomass_time_dataset_3_classes(folder="data/simulated_data/fixed_Max_range_of_parameters/"):
    """

    :param folder: the directory of the datasets
    :return:
    """
    x = "three_classes_classification_without_derivative"
    # noise_type = ["time_dependent_noise_0.2.csv", "time_independent_noise_0.25.csv",
    #                "biomass_dependent_noise_0.2.csv", "without_noise.csv"]
    noise_type = ["time_dependent_noise_0.2", "time_independent_noise_0.25",
                  "biomass_dependent_noise_0.2", "without_noise"]
    X_name = ["simulated_X_data_irradiance_",
            "simulated_X_data_logistic_",
            "simulated_X_data_Allee_"]
    Y_name = ["simulated_label_data_irradiance_",
              "simulated_label_data_logistic_",
              "simulated_label_data_Allee_"]
    cross_validation_result = pd.DataFrame()
    for noise in noise_type:
        X_name_list = [(folder+name + noise) for name in X_name]
        Y_name_list = [(folder+name + noise) for name in Y_name]
        dataset_list = list(zip(X_name_list,Y_name_list))
        print("model training for {} noise".format(noise))
        print(dataset_list)
        x_train, x_test, y_train, y_test, scaler_X_train, scaler_X_test = reformat.combine_data_for_model_classification(
            dataset_list)
        test_label_frame = pd.read_csv("{}_test_label.csv".format(dataset_list[0][0].replace("irradiance","")),header=0,index_col=0)
        for optimizer in ["Adam"]:
            for lr in [0.001,0.01,0.1]:
                for epoch in [300]:
                    for hidden_size in [2,6,10]:
                        for batch_size in [10,32,64]:
                            model = model_classify.LSTMModelClassification(
                                input_size=x_train.shape[2], hidden_size=hidden_size,
                                ouput_size=y_train.shape[1])
                            print("optimizer:{}".format(optimizer))
                            print("lr: {}".format(lr))
                            print("hidden_size:{}".format(hidden_size))
                            print("batch_size:{}".format(batch_size))
                            model, train_accuracy = model_classify.training(
                                model, x=x_train,
                                y=y_train, lr=lr, epoch=epoch, batch_size=batch_size,
                                optimize=optimizer)
                            test_accuracy,predict_label = model_classify.test_model(
                                x_test, y_test, model)
                            print({"noise_type":noise,
                                      "optimizer": optimizer, "lr": lr,
                                      "num_epoch": epoch,
                                      "batch_size": batch_size,
                                      "hidden_size": hidden_size,
                                      "train_accuracy": train_accuracy,
                                      "test_accuracy": test_accuracy})
                            new_row = pd.DataFrame(
                                data={"noise_type":noise,
                                      "optimizer": optimizer, "lr": lr,
                                      "num_epoch": epoch,
                                      "batch_size": batch_size,
                                      "hidden_size": hidden_size,
                                      "train_accuracy": train_accuracy,
                                      "test_accuracy": test_accuracy},
                                index=[0])

                            cross_validation_result = pd.concat(
                                [cross_validation_result, new_row])
                            print(cross_validation_result)
                            test_label_frame["predict_label{}_lr{}_epoch{}_batch{}_hidden{}".format(optimizer,lr,epoch,batch_size,hidden_size)]\
                            = predict_label
                            test_label_frame.to_csv("{}{}_test_label.csv".format(folder,dataset_list[0][0].replace("irradiance","")),header=0,index_col=0)
                            cross_validation_result.to_csv(
                                "{}LSTM{}_cross_validation_result_with_derivative.csv".format(folder,x))

def run_CNN_model_for_image_dataset_3_classes(folder='data/simulated_data/fixed_Max_range_of_parameters/'):

    reformat.read_image_and_reformat(folder)
    x = "three_classes_classification_image"
    noise_type = ["time_dependent_noise_0.2", "time_independent_noise_0.25",
                  "biomass_dependent_noise_0.2"]

    cross_validation_result = pd.DataFrame()
    for noise in noise_type:
        X = dill.load(open("plot/{}_Input_X".format(folder,noise), "rb"))
        Y = dill.load(open("{}plot/{}_Input_label".format(folder,noise), "rb"))
        print("model training for {} noise".format(noise))
        print(Y.shape)

        # add index to Y, use for tracing back
        index = torch.tensor(list(range(Y.shape[0])))
        index = torch.unsqueeze(index,1)

        Y = torch.cat((Y,index),1)

        x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                            test_size=0.2,
                                                            random_state=123)
        #remove index and save seperately
        train_index = y_train[:,3]
        test_index = y_test[:,3]
        y_train = y_train[:,:3]
        y_test = y_test[:,:3]
        label = model_classify.convert_one_hot_endoding_to_label(y_test)
        test_label_frame = pd.DataFrame(data=label,index=list(test_index.numpy()),columns=["label"])
        print(test_label_frame)

        label = model_classify.convert_one_hot_endoding_to_label(y_train)
        train_label_frame = pd.DataFrame(data=label,index=list(train_index.numpy()),columns=["label"])
        print(train_label_frame)

        test_accuracy_old = 0
        for optimizer in ["Adam"]:
            for lr in [0.02,0.01,0.001]:
                for epoch in [30]:
                    for kernel_size in [3,5]:
                        for batch_size in [10,32,64]:
                            for stride in [9,12]:
                                #input_size is a list [input_height,input_width]
                                model = model_classify.CNNModelClassification(
                                    input_channel=x_train.shape[1], kernel_size=kernel_size,
                                    ouput_size=y_train.shape[1],
                                    input_size=[x_train.shape[-2],x_train.shape[-1]],stride=stride)
                                lstm_regression.count_parameters(model)
                                print("optimizer:{}".format(optimizer))
                                print("lr: {}".format(lr))
                                print("kernel_size:{}".format(kernel_size))
                                print("batch_size:{}".format(batch_size))
                                print("stride: {}".format(stride))
                                model, train_accuracy = model_classify.training(
                                    model, x=x_train,
                                    y=y_train, lr=lr, epoch=epoch, batch_size=10,
                                    optimize=optimizer,image=True)
                                test_accuracy,predict_label = model_classify.test_model(
                                    x_test, y_test, model,image=True)
                                if test_accuracy > test_accuracy_old:
                                    torch.save(model.state_dict(), "data/simulated_data/fixed_Max_range_of_parameters/plot/best_model")
                                new_row = pd.DataFrame(
                                    data={"noise_type":noise,
                                          "optimizer": optimizer, "lr": lr,
                                          "num_epoch": epoch,
                                          "batch_size": batch_size,
                                          "kernel_size": kernel_size,
                                          "stride":stride,
                                          "train_accuracy": train_accuracy,
                                          "test_accuracy": test_accuracy},
                                    index=[0])
                                cross_validation_result = pd.concat(
                                    [cross_validation_result, new_row])
                                print(cross_validation_result)
                                test_label_frame["predict_label_{}_{}_lr{}_epoch{}_batch{}_kernel{}_stride{}".format(noise,optimizer,lr,epoch,batch_size,kernel_size,stride)]=predict_label
                                test_label_frame.to_csv("test_result_label.csv")
                                # cross_validation_result.to_csv(
                                #     "{}_cross_validation_result_with_derivative.csv".format(x))

def run_random_forest_model_for_biomass_time_dataset_3_classes():

    noise_type = ["time_dependent_noise_0.2.csv",
                  "time_independent_noise_0.25.csv",
                  "biomass_dependent_noise_0.2.csv", "without_noise.csv"]
    # noise_type = ["time_dependent_noise_0.2", "time_independent_noise_0.25",
    #               "biomass_dependent_noise_0.2", "without_noise"]
    X_name = [
        "data/simulated_data/fixed_Max_range_of_parameters/simulated_smoothed_derivative_data_irradiance_",
        "data/simulated_data/fixed_Max_range_of_parameters/simulated_smoothed_derivative_data_logistic_",
        "data/simulated_data/fixed_Max_range_of_parameters/simulated_smoothed_derivative_data_Allee_"]
    Y_name = [
        "data/simulated_data/fixed_Max_range_of_parameters/simulated_label_data_irradiance_",
        "data/simulated_data/fixed_Max_range_of_parameters/simulated_label_data_logistic_",
        "data/simulated_data/fixed_Max_range_of_parameters/simulated_label_data_Allee_"]
    cross_validation_result = pd.DataFrame()
    for noise in noise_type:
        print("noise:{}".format(noise))
        X_name_list = [(name + noise) for name in X_name]
        Y_name_list = [(name + noise) for name in Y_name]
        input_X = [pd.read_csv(x, header=0, index_col=0) for x in X_name_list]
        input_Y = [pd.read_csv(x, header=0, index_col=0) for x in Y_name_list]

        input = pd.concat(input_X, axis=1)
        Y = pd.concat(input_Y, axis=1).T
        print(input)
        print(Y)
        Y = reformat.one_hot_encoding_for_multiclass_classification(
            torch.tensor(Y.values)).numpy()

        print(np.argmax(Y, axis=1))
        Y = np.argmax(Y, axis=1)
        print(input.iloc[0, :])
        sns.scatterplot(x=input.iloc[0, :], y=input.columns, hue=Y)
        plt.show()
        X_train, X_test, y_train, y_test = train_test_split(input.T.to_numpy(),
                                                            Y, test_size=0.3)
        from model_classify import RandomForest
        rf = RandomForest()

        rf.RF_model(X_train=X_train, y_train=y_train, X_test=X_test,
                    y_test=y_test, file_name="1")

def add_weight_to_labels(class_weight=[0.5, 0.5, 0.25,0.25]):
     #class weight is a list which is the length of classes
     noise_type = ["without_noise","time_dependent_noise_0.2", "time_independent_noise_0.25",
                   "biomass_dependent_noise_0.2"]


     for noise in noise_type:
         X = dill.load(open(
             "data/simulated_data/fixed_Max_range_of_parameters/plot/{}_Input_X".format(
                 noise), "rb"))
         Y = dill.load(open(
             "data/simulated_data/fixed_Max_range_of_parameters/plot/{}_Input_label".format(
                 noise), "rb"))
         print("model training for {} noise".format(noise))
         print(Y.shape)

         # add index to Y, use for tracing back
         index = torch.tensor(list(range(Y.shape[0])))
         index = torch.unsqueeze(index, 1)

         Y = torch.cat((Y, index), 1)

         x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                             test_size=0.2,
                                                             random_state=123)
         # remove index and save seperately
         train_index = y_train[:, 3]
         test_index = y_test[:, 3]
         y_train = y_train[:, :3]
         y_test = y_test[:, :3]
         label = model_classify.convert_one_hot_endoding_to_label(y_test)
         test_label_frame = pd.DataFrame(data=label,
                                         index=list(test_index.numpy()),
                                         columns=["label"])
         print(test_label_frame)

         label = model_classify.convert_one_hot_endoding_to_label(y_train)
         train_label_frame = pd.DataFrame(data=label,
                                          index=list(train_index.numpy()),
                                          columns=["label"])
         print(train_label_frame)


def main():
    # run_lstm_parameters_prediction()
    # x = 'simulated_X_data_test'
    run_LSTM_model_for_biomass_time_dataset_3_classes(folder="data/simulated_data/simulated_from_elope_data_120_no_gene_effect/")
    #run_CNN_model_for_image_dataset_3_classes()
    # tain_loss, test_loss = run_lstm_forcaseting(ratio_of_label=0.5,
    #                                             learning_rate=0.1,
    #                                             num_epoch=300, batch_size=32,
    #                                             hidden_size=6,
    #                                             optimize='Adam',
    #                                             x_name="{}.csv".format(x))
    # cross_validation_result = pd.DataFrame()
    # for optimizer in ["SGD","Adam"]:
    #     for lr in [0.001,0.01,0.1,0.5]:
    #         for epoch in [300]:
    #             for hidden_size in [2,6,10]:
    #                 for ratio in [0.2,0.3,0.4,0.5]:
    #                     for batch_size in [10,32,64]:
    #                         print("optimizer:{}".format(optimizer))
    #                         print("lr: {}".format(lr))
    #                         print("hidden_size:{}".format(hidden_size))
    #                         print("y_ratio:{}".format(ratio))
    #                         print("batch_size:{}".format(batch_size))
    #                         tain_loss,test_loss =run_lstm_forcaseting(ratio_of_label=ratio,learning_rate=lr,
    #                          num_epoch=epoch, batch_size=10, hidden_size=hidden_size,
    #                          optimize=optimizer,x_name="{}.csv".format(x))
    #                         new_row = pd.DataFrame(data={"optimizer":optimizer,"lr":lr,"num_epoch":epoch,"batch_size":batch_size,"hidden_size":hidden_size,"y_ratio":ratio,"train_MSEloss":tain_loss,"test_MSEloss":test_loss},index=[0])
    #                         cross_validation_result = pd.concat([cross_validation_result,new_row])
    #                         print(cross_validation_result)
    #                         cross_validation_result.to_csv("{}_cross_validation_result.csv".format(x))

if __name__ == '__main__':
    main()