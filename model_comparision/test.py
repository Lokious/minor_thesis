import matplotlib.pyplot as plt
import torch

# import from current directory
import lstm_regression
import model_classify
import reformat
import numpy as np
import sys
import os
import pandas as pd
from torch import nn

from sklearn.model_selection import train_test_split

def run_LSTM_model_for_biomass_time_dataset_4_classes(folder="data/simulated_data/fixed_Max_range_of_parameters/", weighted=None,x = "three_classes_classification_without_derivative"):
    """

    :param folder: the directory of the datasets
    :return:
    """

    reformat.save_combined_feature_tensors_as_dill_files(folder)
    # noise_type = ["time_dependent_noise_0.2.csv", "time_independent_noise_0.25.csv",
    #                "biomass_dependent_noise_0.2.csv", "without_noise.csv"]
    noise_type = ["time_dependent_noise_0.2", "time_independent_noise_0.25",
                  "biomass_dependent_noise_0.2", "without_noise"]
    if weighted:
        class_weight = weighted
    # X_name = ["simulated_X_data_irradiance_",
    #         "simulated_X_data_logistic_",
    #         "simulated_X_data_Allee_"]
    # Y_name = ["simulated_label_data_irradiance_",
    #           "simulated_label_data_logistic_",
    #           "simulated_label_data_Allee_"]
    X_name = ["simulated_X_data_irradiance_",
            "simulated_X_data_logistic_",
            "simulated_X_data_Allee_",
            "simulated_X_data_Temperature_"]
    Y_name = ["simulated_label_data_irradiance_",
              "simulated_label_data_logistic_",
              "simulated_label_data_Allee_",
              "simulated_label_data_Temperature_"]
    cross_validation_result = pd.DataFrame()
    for noise in noise_type:
        X_name_list = [(folder+name + noise) for name in X_name]
        Y_name_list = [(folder+name + noise) for name in Y_name]
        dataset_list = list(zip(X_name_list,Y_name_list))
        print("model training for {} noise".format(noise))
        print(dataset_list)
        x_train, x_test, y_train, y_test, scaler_X_train, scaler_X_test = reformat.combine_data_for_model_classification(
            dataset_list,test_size=1/30,train_size=2/15) #use the same number of data as other datasets
        test_label_frame = pd.read_csv("{}_test_label.csv".format(dataset_list[0][0].replace("irradiance","")),header=0,index_col=0)
        for optimizer in ["Adam"]:
            for lr in [0.005,0.01,0.001]:
                for epoch in [300]:
                    for hidden_size in [2,6,10]:
                        for batch_size in [10,32,64]:
                            model = model_classify.LSTMModelClassification(
                                input_size=x_train.shape[2], hidden_size=hidden_size,
                                output_size=y_train.shape[1])
                            print("optimizer:{}".format(optimizer))
                            print("lr: {}".format(lr))
                            print("hidden_size:{}".format(hidden_size))
                            print("batch_size:{}".format(batch_size))
                            if weighted:
                                print("weighted lass")
                                print(class_weight)
                                model, train_accuracy = model_classify.training(
                                    model, x=x_train,
                                    y=y_train, lr=lr, epoch=epoch, batch_size=batch_size,
                                    optimize=optimizer,weight=class_weight)
                            else:
                                model, train_accuracy = model_classify.training(
                                    model, x=x_train,
                                    y=y_train, lr=lr, epoch=epoch, batch_size=batch_size,
                                    optimize=optimizer)
                            test_accuracy,predict_label,avg_auc_score = model_classify.test_model(
                                x_test, y_test, model)
                            print({"noise_type":noise,
                                      "optimizer": optimizer, "lr": lr,
                                      "num_epoch": epoch,
                                      "batch_size": batch_size,
                                      "hidden_size": hidden_size,
                                      "train_accuracy": train_accuracy,
                                      "test_accuracy": test_accuracy,
                                   "avg_auc_score":avg_auc_score})
                            new_row = pd.DataFrame(
                                data={"noise_type":noise,
                                      "optimizer": optimizer, "lr": lr,
                                      "num_epoch": epoch,
                                      "batch_size": batch_size,
                                      "hidden_size": hidden_size,
                                      "train_accuracy": train_accuracy,
                                      "test_accuracy": test_accuracy,
                                      "avg_auc_score":avg_auc_score},
                                index=[0])

                            cross_validation_result = pd.concat(
                                [cross_validation_result, new_row])
                            print(cross_validation_result)
                            test_label_frame["predict_label{}_lr{}_epoch{}_batch{}_hidden{}".format(optimizer,lr,epoch,batch_size,hidden_size)]\
                            = predict_label
                            test_label_frame.to_csv("{}_test_label_weighted.csv".format(dataset_list[0][0].replace("irradiance","")))
                            if weighted:
                                cross_validation_result.to_csv(
                                "{}LSTM{}_cross_validation_result_with_derivative_4_classes_weighted.csv".format(folder,x))
                            else:
                                cross_validation_result.to_csv(
                                    "{}LSTM{}_cross_validation_result_with_derivative_4_classes.csv".format(
                                        folder, x))

def predict_simulated_snps(folder="data/simulated_data/simulated_with_different_gene_type/reformat_data/"):
    noise_type = ["time_dependent_noise_0.2", "time_independent_noise_0.25",
                  "biomass_dependent_noise_0.2", "without_noise"]

    for noise in noise_type:
        #read x growth curve and derivative

        input_biomass = "data/simulated_data/simulated_with_different_gene_type/reformat_data/{}_Input_X.csv".format(noise)
        input_derivate = "data/simulated_data/simulated_with_different_gene_type/reformat_data/{}_Input_derivative.csv".format(noise)
        #read snps as label y
        y = "data/simulated_data/simulated_with_different_gene_type/reformat_data/{}_Input_snp.csv".format(noise)
        X_tensor1, tensor_lable = reformat.read_df_create_input_data(input_biomass, y,
                                                                   random_select=False)
        X_tensor2, tensor_lable = reformat.read_df_create_input_data(input_derivate, y,
                                                                   random_select=False)

        X_tensor= torch.concat([X_tensor1,X_tensor2],dim=-1)

        for snp_index in range(4):

            test_index, x_train,x_test, y_test, y_train,label = perfrom_train_test_split_for_snps_and_save_test_index(
                X_tensor, snp_index, tensor_lable)
            test_label_frame = pd.DataFrame(data=label,
                                            index=list(test_index.numpy()),
                                            columns=["snp_{}".format(snp_index+1)])
            print(test_label_frame)
            print(x_train.shape,y_train.shape)
            cross_validation_result = pd.DataFrame()
            for optimizer in ["Adam"]:
                for lr in [0.005, 0.01, 0.001]:
                    for epoch in [300]:
                        for hidden_size in [2, 6, 10]:
                            for batch_size in [10, 32, 64]:
                                model = model_classify.LSTM_snp_classification(
                                    input_size=x_train.shape[2],
                                    hidden_size=hidden_size)
                                print("optimizer:{}".format(optimizer))
                                print("lr: {}".format(lr))
                                print("hidden_size:{}".format(hidden_size))
                                print("batch_size:{}".format(batch_size))
                                model, train_accuracy = model_classify.training(
                                    model, x=x_train,
                                    y=y_train, lr=lr, epoch=epoch,
                                    batch_size=batch_size,
                                    optimize=optimizer)
                                test_accuracy, predict_label, avg_auc_score = model_classify.test_model(
                                    x_test, y_test, model)
                                print({"noise_type": noise,
                                       "optimizer": optimizer, "lr": lr,
                                       "num_epoch": epoch,
                                       "batch_size": batch_size,
                                       "hidden_size": hidden_size,
                                       "train_accuracy": train_accuracy,
                                       "test_accuracy": test_accuracy,
                                       "avg_auc_score": avg_auc_score})
                                new_row = pd.DataFrame(
                                    data={"noise_type": noise,
                                          "optimizer": optimizer, "lr": lr,
                                          "num_epoch": epoch,
                                          "batch_size": batch_size,
                                          "hidden_size": hidden_size,
                                          "train_accuracy": train_accuracy,
                                          "test_accuracy": test_accuracy,
                                          "avg_auc_score": avg_auc_score},
                                    index=[0])

                                cross_validation_result = pd.concat(
                                    [cross_validation_result, new_row])
                                print(cross_validation_result)
                                test_label_frame[
                                    "predict_label{}_lr{}_epoch{}_batch{}_hidden{}".format(
                                        optimizer, lr, epoch, batch_size,
                                        hidden_size)] \
                                    = predict_label
                                test_label_frame.to_csv(
                                    "{}_test_snps.csv".format("data/simulated_data/simulated_with_different_gene_type/reformat_data/"))

                                cross_validation_result.to_csv(
                                    "{}LSTM_snp{}_cross_validation_result_with_derivative_.csv".format(
                                        folder,snp_index+1 ))

def perfrom_train_test_split_for_snps_and_save_test_index(X_tensor, snp_index,
                                                          tensor_lable):
    tensor_lable = tensor_lable[:, snp_index]

    #one hot encoding
    Y_tensor = reformat.one_hot_encoding_for_multiclass_classification(
        tensor_lable)
    print(Y_tensor.shape)
    num_classes = Y_tensor.shape[1]
    index = torch.tensor(list(range(Y_tensor.shape[0])))
    index = torch.unsqueeze(index, 1)
    print(index.shape)
    Y_tensor = torch.cat((Y_tensor, index), 1)
    print("Y tensor")
    print(Y_tensor.shape)


    # print("Y tensor add index")
    # print(Y_tensor)
    x_train, x_test, y_train, y_test = train_test_split(X_tensor, Y_tensor,
                                                        random_state=123
                                                        , test_size=1 / 30,
                                                        train_size=2 / 15)

    # remove index and save seperately
    train_index = y_train[:, num_classes]
    test_index = y_test[:, num_classes]
    y_train = y_train[:, :num_classes]
    y_test = y_test[:, :num_classes]
    label = model_classify.convert_one_hot_endoding_to_label(y_test)
    return test_index, x_train,x_test, y_test, y_train,label


def main():
    predict_simulated_snps()
    #run_LSTM_model_for_biomass_time_dataset_4_classes(folder="data/simulated_data/simulated_from_elope_data_120_no_gene_effect/",weighted=[0.5, 0.5, 0.5, 0.25])
    # run_LSTM_model_for_biomass_time_dataset_4_classes(folder
    #  ="data/simulated_data/simulated_with_different_gene_type/reformat_data/",x="withe_gene_effect")
if __name__ == '__main__':
    main()