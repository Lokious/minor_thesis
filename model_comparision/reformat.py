import pandas as pd
import numpy as np
import glob
import dill
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
def main():
    #read_reformat("./data/")
    # for name in ["Merredin","Narrabri","Yanco"]:
    #     file_name="data/{}_data.csv".format(name)
    #     read_and_reformat_for_clustering(file_name)
    X,Y =read_df_create_input_data(inputX="simulated_X_data_1.csv", inputY="simulated_Y_data_1.csv")
    X, scaler = lstm_regression.normalization(X)
    print(X)
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=123)
    # define model

    model = lstm_regression.LSTM_paramter_regression(input_size=1, hidden_size=2, dropout=0.01)
    # model training
    print(model)
    model = lstm_regression.training(model, x=x_train, y=y_train, batch_size=10)

    # test model

    lstm_regression.test_model(x_test, y_test, model)
if __name__ == '__main__':
    main()
    