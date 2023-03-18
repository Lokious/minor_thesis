import numpy
import pandas as pd
import numpy as np
import glob
import dill
import unittest
import torch
from sklearn.model_selection import train_test_split
import lstm_regression
def read_reformat(file_directory:str,file_name_end_with=".txt"):
    files = glob.glob("{}*{}".format(file_directory,file_name_end_with))
    print(files)
    biomass_df =pd.DataFrame()
    # read and merge to one csv
    for file in files:
        df_new = pd.read_table(file,header=0,index_col=0,sep=" ")
        biomass_df = pd.concat([biomass_df,df_new])
        print(biomass_df)
    else:
        biomass_df.to_csv("biomass.csv")

    #separate based on environment
    biomass_df.groupby("Env")

import copy

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.cluster import KMeans
import os

def read_and_reformat_for_clustering(file:str="data/Emerald_data.csv"):
    input_data = pd.read_csv(file,header=0,index_col=0)
    loc = input_data["loc"].unique()[0]
    groups_object = input_data.groupby(['Env','geno'])
    group_number = len(groups_object.groups)
    print("group_num:{}".format(group_number))
    length = 0
    time_list = []
    plantid_list = []
    for i,item in enumerate(groups_object.groups):

        # group based on genotype and location
        plant_df = groups_object.get_group(item)
        plant_df.set_index("das",inplace=True)
        #print(plant_df)
        time_length = len(plant_df.index)
        if time_length > length:
            length = time_length
            time_list = list(plant_df.index)
        plantid_list.append((str(plant_df["Env"].unique()[0])+"."+str(plant_df["geno"].unique()[0])+"."+str(i)))
    #print(length)


    new_df_Biomass = pd.DataFrame(index=time_list, columns=plantid_list)

    input_data.set_index(['Env','geno','das'],inplace=True)

    print(input_data)
    for i in time_list:
        print(i)
        for column in plantid_list:
            env_geno=column.split(".")[:-1]
            try:
                new_df_Biomass.loc[i,column] = input_data.loc[(env_geno[0],env_geno[1], i),"biomass"]
                #print(input_data.loc[(env_geno[0],env_geno[1], i),"biomass"])
            except:
                #raise ValueError("nan")
                new_df_Biomass.loc[i, column] = np.nan

    print(new_df_Biomass)
    new_df_Biomass.to_csv("data/df_Biomass_{}.csv".format(loc))
    with open("reformat_biomass_{}".format(loc),"wb") as dillfile:
        dill.dump(new_df_Biomass,dillfile)
    return new_df_Biomass


def read_df_create_input_data(inputX,inputY):
    """
    This function is to read the X.csv and y.csv and convert it to torch.tensor
    :param inputX: the name and directory of x
    :param inputY: the name and directory of y
    :return: tensor x and y
    """
    X_df = pd.read_csv(inputX,index_col=0)
    Y_df = pd.read_csv(inputY,index_col=0)
    print(X_df)
    print(Y_df)
    # the column is related to number of samples, which should
    assert len(X_df.columns) == len(Y_df.columns)
    Y_df.columns = X_df.columns
    drop_na_column = X_df.columns[X_df.isna().any()].tolist()
    X_df.dropna(axis=1,inplace=True)
    Y_df = Y_df.drop(columns=drop_na_column)
    Y_df = Y_df.T
    #print(X_df.isna().sum().sum())

    import torch
    sequences = X_df.astype(np.float32).to_numpy().tolist()
    X_tensor = torch.stack(
        [torch.tensor(s).unsqueeze(1).float() for s in sequences])#[118, 101, 1]
    X_tensor = torch.permute(X_tensor, (1,0,2)) # (n_seq,seq_len,n_features)
    print(X_tensor.shape)

    # creating tensor from targets_df
    tensor_lable = torch.tensor(Y_df.values.astype('float'))#[1,101]
    print(tensor_lable.shape)

    return X_tensor,tensor_lable


def prepatre_train_test_for_lstm_regression(x="simulated_X_data_3.csv",y="simulated_Y_data_3.csv"):

    import lstm_regression
    X,Y =read_df_create_input_data(inputX=x, inputY=y)

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=123)

    x_train, scaler_X_train = lstm_regression.normalization(x_train)
    x_test, scaler_X_test = lstm_regression.normalization(x_test)
    print(X)
    return x_train, x_test, y_train, y_test


def prepare_train_test_for_lstm_forecasting_multiple_dimension_label(x="simulated_X_data_2.csv", label_size=0.6,test_size=0.2):

    #read time series df
    X_df = pd.read_csv(x,index_col=0)
    if x.startswith("simulated"):
        X_df.dropna(axis=1,inplace=True)
    else:
        X_df.dropna(axis=0, inplace=True)
    print(X_df)
    # assert no NA left
    assert X_df.isna().sum().sum() == 0
    # plt.plot(X_df.iloc[:,2])
    # plt.show()
    # split into train and test

    time_steps = len(X_df.index)
    test_length = int(time_steps * label_size)
    print(test_length)
    train_length = time_steps - test_length

    Y_df = X_df.iloc[train_length:,:]
    X_df = X_df.iloc[0:train_length-1,:]
    #print(Y_df)
    sequences_X = X_df.astype(np.float32).to_numpy().tolist()
    X_tensor = torch.stack(
        [torch.tensor(s).unsqueeze(1).float() for s in sequences_X])#[118, 101, 1]
    X_tensor = torch.permute(X_tensor, (1,0,2)) # (n_seq,seq_len,n_features)
    sequences_Y = Y_df.astype(np.float32).to_numpy().tolist()
    Y_tensor = torch.stack(
        [torch.tensor(s).float() for s in sequences_Y])#(seq_len,n_seq)
    print(Y_tensor.shape)
    Y_tensor = torch.permute(Y_tensor, (1,0)) # (n_seq,seq_len)
    print(Y_tensor.shape)

    #X_tensor is the first 70% time step and Y_tensor is following 30% times steps datapoints
    #return X_tensor,Y_tensor

    x_train, x_test, y_train, y_test = train_test_split(X_tensor, Y_tensor,
                                                        test_size=test_size,
                                                        random_state=123)

    x_train, scaler_X_train = lstm_regression.normalization(x_train)
    x_test, scaler_X_test = lstm_regression.normalization(x_test)
    #save for test scaler
    y_train_unscale = copy.deepcopy(y_train)

    from sklearn.preprocessing import StandardScaler
    scaler_Y_train = StandardScaler()
    scaler_Y_train = scaler_Y_train.fit(y_train)
    # normalize the dataset and print
    standardized = scaler_Y_train.transform(y_train)
    # print(standardized.shape)
    y_train = torch.stack([torch.tensor(s).float() for s in standardized])
    print(y_train.shape)
    # sacler for y
    scaler_Y_test = StandardScaler()
    scaler_Y_test = scaler_Y_test.fit(y_test)
    # normalize the dataset and print
    standardized = scaler_Y_test.transform(y_test)
    # print(standardized.shape)
    y_test = torch.stack([torch.tensor(s).float() for s in standardized])
    print(y_test.shape)

    #float will return error, sonver to int
    check_scaler(scaler_Y_train, y_train, y_train_unscale)
    return x_train, x_test, y_train, y_test, scaler_Y_train,scaler_X_train,scaler_Y_test,scaler_X_test,y_train_unscale

def prepare_train_test_for_lstm_forecasting_one_dimension_label(x="simulated_X_data_1.csv", label_size=0.3,test_size=0.2):

    #read time series df
    X_df = pd.read_csv(x,index_col=0)
    #X_df.dropna(axis=1,inplace=True)
    X_df.dropna(axis=0, inplace=True)
    #print(X_df)
    # assert no NA left
    assert X_df.isna().sum().sum() == 0
    plt.plot(X_df.iloc[:,0])
    plt.show()
    # split into train and test

    time_steps = len(X_df.index)
    test_length = int(time_steps * label_size)
    print("test_length{}".format(test_length))
    train_length = time_steps - test_length

    Y_df = X_df.iloc[train_length:,:]
    X_df = X_df.iloc[0:train_length-1,:]
    print(Y_df)
    sequences_X = X_df.astype(np.float32).to_numpy().tolist()
    X_tensor = torch.stack(
        [torch.tensor(s).unsqueeze(1).float() for s in sequences_X])#[118, 101, 1]
    X_tensor = torch.permute(X_tensor, (1,0,2)) # (n_seq,seq_len,n_features)
    sequences_Y = Y_df.astype(np.float32).to_numpy().tolist()
    Y_tensor = torch.stack(
        [torch.tensor(s).float() for s in sequences_Y])#(seq_len,n_seq)
    print(Y_tensor.shape)
    Y_tensor = torch.permute(Y_tensor, (1,0)) # (n_seq,seq_len)
    print(Y_tensor.shape)

    #X_tensor is the first 70% time step and Y_tensor is following 30% times steps datapoints
    #return X_tensor,Y_tensor

    x_train, x_test, y_train, y_test = train_test_split(X_tensor, Y_tensor,
                                                        test_size=test_size,
                                                        random_state=123)

    x_train, scaler_X_train = lstm_regression.normalization(x_train)
    x_test, scaler_X_test = lstm_regression.normalization(x_test)

    from sklearn.preprocessing import StandardScaler
    scaler_Y_train = StandardScaler()
    scaler_Y_train = scaler_Y_train.fit(y_train)
    # normalize the dataset and print
    standardized = scaler_Y_train.transform(y_train)
    # print(standardized.shape)
    y_train = torch.stack([torch.tensor(s).float() for s in standardized])
    print(y_train.shape)
    # sacler for y
    scaler_Y_test = StandardScaler()
    scaler_Y_test = scaler_Y_test.fit(y_train)
    # normalize the dataset and print
    standardized = scaler_Y_test.transform(y_train)
    # print(standardized.shape)
    y_test = torch.stack([torch.tensor(s).float() for s in standardized])
    print(y_test.shape)

    return x_train, x_test, y_train, y_test

def check_scaler(scaler, scaled_data:numpy.array, original_data:torch.tensor):

    unscale = scaler.inverse_transform(scaled_data)
    unscale = torch.stack(
        [torch.tensor(s).float() for s in unscale])
    #float will return error, sonver to int
    print("###test if unsalced data is equal to the inverse transfer data###")
    print("raise error if False check")
    # print(torch.eq(unscale, original_data))
    # print(torch.sum(torch.eq(unscale, original_data)).item() / original_data.nelement())
    assert torch.sum(torch.eq(unscale,
                              unscale)).item() / original_data.nelement() == 1.0
def main():
    prepare_train_test_for_lstm_forecasting_multiple_dimension_label(x="simulated_X_data_2.csv",
                                            label_size=0.2)
    #unittest.main()
    #read_reformat("./data/")
    # for name in ["Merredin","Narrabri","Yanco"]:
    #     file_name="data/{}_data.csv".format(name)
    #     read_and_reformat_for_clustering(file_name)

if __name__ == '__main__':
    main()
    