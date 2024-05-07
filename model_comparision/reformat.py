import random

import pandas as pd
import numpy as np
import glob
import dill
import cv2
import os
import unittest
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from statsmodels.stats.anova import AnovaRM

import lstm_regression
from PIL import Image

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


def read_df_create_input_data(inputX,inputY,random_select=False):
    """
    This function is to read the X.csv and y.csv and convert it to torch.tensor
    :param inputX: the name and directory of x
    :param inputY: the name and directory of y
    :return: tensor x and y x.shape(n_seq,seq_len,n_features) y.shape(feature,n_seq)
    """
    X_df = pd.read_csv(inputX,index_col=0)
    Y_df = pd.read_csv(inputY,index_col=0)
    try:
        Y_df = pd.DataFrame(Y_df.loc["r",:])
    except:
        print("no column name")
    print(X_df,Y_df)
    # the column is related to number of samples, which should be the same length
    try:
        assert len(X_df.columns) == len(Y_df.columns)
    except:
        Y_df = Y_df.T
        assert len(X_df.columns) == len(Y_df.columns)
    Y_df.columns = X_df.columns
    #replace inf and -inf as NA
    X_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    #drop na
    drop_na_column = X_df.columns[X_df.isna().any()].tolist()
    X_df.dropna(axis=1,inplace=True)
    Y_df = Y_df.drop(columns=drop_na_column)

    Y_df = Y_df.T
    #print(X_df.isna().sum().sum())
    if random_select:
        X_df = X_df.iloc[:, :200]
        Y_df = Y_df.iloc[:200, :]


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

    x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=123)

    x_train, scaler_X_train = lstm_regression.normalization(x_train)
    x_test, scaler_X_test = lstm_regression.normalization(x_test)
    #print(X)
    return x_train, x_test, y_train, y_test


def prepare_train_test_for_lstm_forecasting_multiple_dimension_label(x="simulated_X_data_2.csv", label_size=0.6,test_size=0.2):

    #read time series df
    X_df = pd.read_csv(x,index_col=0)
    #print(X_df)
    if x.startswith("simulated"):
        X_df.dropna(axis=1,inplace=True)
    else:
        #fill na at the start with 0, and fill na at the end as maximum
        X_df = X_df.fillna(method='ffill')
        #print(X_df.T.isna().sum())
        X_df = X_df.fillna(0.0)

        #X_df.dropna(axis=1, inplace=True)
    #print(X_df)
    # assert no NA left
    assert X_df.isna().sum().sum() == 0
    # plt.plot(X_df.iloc[:,2])
    # plt.show()
    # split into train and test

    time_steps = len(X_df.index)
    test_length = int(time_steps * label_size)
    print(test_length)
    train_length = time_steps - test_length
    #Y_df = X_df
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
    #check_scaler(scaler_Y_train, y_train, y_train_unscale)
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

def check_scaler(scalers, scaled_data:np.array, original_data:torch.tensor):

    try:
        scaler = scalers[0]
        inverse_scale = scaler.inverse_transform(scaled_data)
        inverse_scale = torch.stack(
            [torch.tensor(s).float() for s in inverse_scale])

        print("###test if unsalced data is equal to the inverse transfer data###")
        print("raise error if False check")
        assert torch.sum(torch.eq(inverse_scale,
                                  original_data)).item() / original_data.nelement() == 1.0
    except:
        inverse_scale_list = []
        dataset = torch.permute(scaled_data, (2, 1, 0))
        tensors = list(dataset)

        for i,feature_tensor in enumerate(tensors):
            print(feature_tensor.shape)
            scaler = scalers[i]
            #perform inverse_scale for each feature in X
            inverse_scale = scaler.inverse_transform(feature_tensor.detach().cpu().numpy())
            inverse_scale = torch.stack(
                [torch.tensor(s).float() for s in inverse_scale])
            print(inverse_scale)
            inverse_scale_list.append(inverse_scale)
        print(
            "###test if unsalced data is equal to the inverse transfer data###")
        print("raise error if False check")
        inverse_scale = torch.stack(inverse_scale_list)
        unscaled_dataset = torch.permute(inverse_scale, (2, 1, 0))
        # the assert error occure because of float

        print(torch.sum(torch.eq(unscaled_dataset, original_data)).item() / original_data.nelement())
        #assert torch.sum(torch.eq(unscaled_dataset,original_data)).item() / original_data.nelement() == 1.0

def combine_data_for_model_classification(datas=[("simulated_X_data_irradiance.csv", "simulated_label_data_irradiance.csv")], test_size=0.2,train_size=0.8,random_select=False,snp_convert=False,seed = 123):

    try:
        X = []
        Y = []
        for data_name in datas:
            """ loop through datasets generate by different SDE model"""
            x_name = data_name[0]
            y_name = data_name[1]
            X_tensor = dill.load(open(x_name,"rb"))

            print("X shape before combine data from different sde{}".format(X_tensor.shape))
            tensor_lable = dill.load(open(y_name,"rb"))
            print(tensor_lable.shape)
            if random_select:
                indexes = random.sample(range(X_tensor.shape[0]),200)
            else:
                indexes = range(X_tensor.shape[0])
            X_tensor = X_tensor[indexes,:,:]
            tensor_lable = tensor_lable[:,indexes]
            tensor_lable = torch.permute(tensor_lable,(1,0))
            print(X_tensor.shape)

            X.append(X_tensor)
            Y.append(tensor_lable)
    except:
        print("inputs are csv file")
        X = []
        Y = []
        for data_name in datas:
            """ loop through datasets generate by different SDE model"""
            x_name = data_name[0]
            y_name = data_name[1]
            X_tensor, tensor_lable = read_df_create_input_data(x_name, y_name,
                                                               random_select=False)
            X.append(X_tensor)
            Y.append(tensor_lable)
    # concat tensor from three models
    X_tensor = torch.cat(X, 0)
    Y_tensor = torch.cat(Y, 0)

    Y_tensor = one_hot_encoding_for_multiclass_classification(Y_tensor)
    # print(X_tensor.shape)
    # print(Y_tensor.shape)
    num_classes = Y_tensor.shape[1]
    # add index to Y, use for tracing back
    index = torch.tensor(list(range(Y_tensor.shape[0])))
    index = torch.unsqueeze(index, 1)
    print(index.shape)
    Y_tensor = torch.cat((Y_tensor, index), 1)
    print("Y tensor")
    print(Y_tensor)

    x_train, x_test, y_train, y_test = train_test_split(X_tensor, Y_tensor,
                                                        test_size=test_size,
                                                        random_state=seed,train_size=train_size)
    import model_classify
    # remove index and save seperately
    train_index = y_train[:, num_classes]
    test_index = y_test[:, num_classes]
    y_train = y_train[:, :num_classes]
    y_test = y_test[:, :num_classes]
    label = model_classify.convert_one_hot_endoding_to_label(y_test)
    test_label_frame = pd.DataFrame(data=label, index=list(test_index.numpy()),
                                    columns=["label"])
    test_label_frame.to_csv("{}_test_label.csv".format(datas[0][0].replace("irradiance","")))

    label = model_classify.convert_one_hot_endoding_to_label(y_train)
    train_label_frame = pd.DataFrame(data=label,
                                     index=list(train_index.numpy()),
                                     columns=["label"])
    print(train_label_frame)
    train_label_frame.to_csv("{}_train_label.csv".format(datas[0][0].replace("irradiance","")))

    # save for test scaler
    x_train_unscale = copy.deepcopy(x_train)

    x_train, scaler_X_train = lstm_regression.normalization(x_train)
    x_test, scaler_X_test = lstm_regression.normalization(x_test)
    # float will return error, sonver to int
    check_scaler(scaler_X_train, x_train, x_train_unscale)
    print(x_train,y_train)
    return x_train, x_test, y_train, y_test,  scaler_X_train, scaler_X_test

def one_hot_encoding_for_multiclass_classification(Y_tensor):

    #check the number of label
    label_number = len(torch.unique(Y_tensor))
    print("number of classes:{}".format(label_number))
    one_hot_list = []
    for y in Y_tensor:
        #covert to int use as index
        y=int(y.item())
        one_hot_code = torch.zeros(label_number)
        one_hot_code[y] = 1
        one_hot_list.append(one_hot_code)
    #print(one_hot_list)
    #torch stack increase the demension, while torch cat does not
    one_hot_encoding = torch.stack(one_hot_list)

    return one_hot_encoding


def create_tensor_dataset(dfs):
    """
    create 3-dimensions tensor from a list of dataframes
    :param dfs: list, the row of df is sequence length, the columns are samples
    :return: X_shape(features,seq_length,samples_number);Y_shape(1,samples_number)
    """

    datasets = []
    for df in dfs:
        sequences = df.astype(np.float32).to_numpy().tolist()
        dataset = torch.stack([torch.tensor(s).unsqueeze(1).float() for s in sequences])
        datasets.append(dataset)

    #remove the last dimention
    tensor_dataset = torch.squeeze(torch.stack(datasets,dim=0))
    # print("shape of created dataset:")
    # print(tensor_dataset.shape)

    n_features, seq_len, n_seq = tensor_dataset.shape
    print("seq_len{}".format(seq_len))
    return tensor_dataset,n_features

def combine_multiple_features_to_inputX(input_features_files:list,input_Y_file:str,save_directory:str,x_name:str,noise_type:str):
    """
    read csv file,
    :param input_features_files:
    :param save_directory:
    :param x_name:
    :return:
    """
    dfs = []
    for feature_file in input_features_files:
        df = pd.read_csv(feature_file,header=0,index_col=0)
        dfs.append(df)

    tensor_X, n_features = create_tensor_dataset(dfs)
    print(tensor_X.shape)
    # creating tensor from targets_df
    label_df = pd.read_csv(input_Y_file,index_col=0,header=0)
    tensor_lable = torch.tensor(label_df.values.astype('double'))

    print(tensor_lable.shape)
    # the number represent the number in old order, and after that will place it
    # as the new order for example: permute(2,0,1)
    # (a,b,c) -> (c,a,b)
    tensor_X = torch.permute(tensor_X, (2, 1, 0)) # (n_seq,seq_len,n_features)

    #creat folder to save input files
    isExist = os.path.exists("{}/".format(save_directory))
    if not isExist:
        print("create directory {}".format(save_directory))
        os.makedirs("{}/".format(save_directory))
    print("saving files...")
    with open("{}/simulated_X_data_{}_{}".format(save_directory,x_name,noise_type),"wb") as dillfile1:
        dill.dump(tensor_X,dillfile1)
    with open("{}/simulated_label_data_{}_{}".format(save_directory,x_name,noise_type),"wb") as dillfile2:
        dill.dump(tensor_lable,dillfile2)

    return tensor_X,tensor_lable,n_features


def save_combined_feature_tensors_as_dill_files(folder="data/simulated_data/fixed_Max_range_of_parameters/"):
    """
    Combine biomass and derivative and save as dill file for each model

    """
    noises = ["time_dependent_noise_0.2.csv",
              "time_independent_noise_0.25.csv",
              "biomass_dependent_noise_0.2.csv", "without_noise.csv"]
    models = ["irradiance_", "logistic_", "Allee_","Temperature_"]
    input_X_list = [
        "simulated_X_data_",
        "simulated_derivative_data_"]
    Y_name = "simulated_label_data_"
    for noise in noises:
        for model in models:
            input_X_list_new = [(folder+x + model + noise) for x in input_X_list]
            Y_name_new = folder+Y_name + model + noise
            combine_multiple_features_to_inputX(
                input_features_files=input_X_list_new, input_Y_file=Y_name_new,
                save_directory=folder,
                x_name=model.replace("_", ""), noise_type=noise.split(".csv")[0])


def read_image_and_reformat(folder="data/simulated_data/fixed_Max_range_of_parameters/"):
    """
    reformat and stack the images to shape:(n_samples, depth, height, width)
    """
    # Define the label for each image class
    label_dict = {'logistic': 0, 'irradiance': 1, 'Allee': 2,'Temperature':3}
    models = ["irradiance", "logistic", "Allee","Temperature"]
    for noise_name in ['time_independent_noise_0.25', 'time_dependent_noise_0.2', 'biomass_dependent_noise_0.2','without_noise']:
        # Initialize the dataset arrays
        data = []
        labels = []
        for sde_model in models:
            for i in range(1,301):
                growth_curve_image_file = "{}plot/growth_curve/simulated_X_data_{}_{}_{}_.tiff".format(folder,sde_model,noise_name,i)
                derivative_image_file = "{}plot/smooth_derivative/simulated_X_data_{}_{}_{}_.tiff".format(folder,sde_model,noise_name,i)

                # read growth curve
                growth_curve_array = cv2.imread(growth_curve_image_file)
                # Convert the image to grayscale
                gray_growth_curve_array = cv2.cvtColor(growth_curve_array, cv2.COLOR_BGR2GRAY)
                gray_growth_curve_array = cv2.normalize(gray_growth_curve_array, None, 0, 1.0,
                                               cv2.NORM_MINMAX,
                                               dtype=cv2.CV_32F)

                #print(gray_growth_curve_array.shape)
                # read derivative
                derivative_array = cv2.imread(derivative_image_file)
                # Convert the image to grayscale
                gray_derivative_array = cv2.cvtColor(derivative_array, cv2.COLOR_BGR2GRAY)
                gray_derivative_array = cv2.normalize(gray_derivative_array, None, 0, 1.0,
                                               cv2.NORM_MINMAX,
                                               dtype=cv2.CV_32F)

                # cv2.imshow("ds", gray_derivative_array)
                # cv2.waitKey(0)
                # stack growth curve and derivative
                input_images = np.stack([gray_growth_curve_array,gray_derivative_array],axis=0)
                #print(input_images.shape)
                # Add the image and its label to the dataset arrays
                data.append([input_images])
                labels.append(label_dict[sde_model])
        else:
            data = torch.tensor(np.array(data).squeeze()) # shape (n_samples,n_features,height,width)
            labels = torch.tensor(np.array(labels)) #(n_samples)
            print("data shape:{}".format(data.shape))
            #one hot encoding for label
            labels = one_hot_encoding_for_multiclass_classification(labels)
            print(labels.shape)

            # save input as dill file
            with open("{}plot/{}_Input_X".format(folder,noise_name), "wb") as dillfile:
                dill.dump(data, dillfile)
            torch.save(data,"{}plot/{}_Input_X.pt".format(folder,noise_name))
            with open("{}plot/{}_Input_label".format(folder,noise_name), "wb") as dillfile:
                dill.dump(labels, dillfile)
            torch.save(labels,"{}plot/{}_Input_label.pt".format(folder,noise_name))


def read_image_and_reformat_for_snps_prediction(directory="data/simulated_data/simulated_with_different_gene_type/plot/"):
    """
    reformat and stack the images to shape:(n_samples, depth, height, width)
    """
    models = ["irradiance", "logistic", "Allee", "Temperature"]
    noises = ['time_independent_noise_0.25', 'time_dependent_noise_0.2',
              'biomass_dependent_noise_0.2', 'without_noise']
    label_dict = {'logistic': 0, 'irradiance': 1, 'Allee': 2, 'Temperature': 3}
    for noise_type in noises:
        snp_label = pd.DataFrame()
        sde_model_type_label = pd.DataFrame()
        image_np_list = []
        for model_type in models:
            # example file name: simulated_X_data_Allee_biomass_dependent_noise_0.2_0_0_0_0_2_.tiff
            x_files = glob.glob(
                "{}growth_curve/simulated_X_data_{}_{}_*{}".format(directory, model_type,
                                                      noise_type, ".tiff"))
            derivative_files = glob.glob(
                "{}smooth_derivative/simulated_X_data_{}_{}_*{}".format(directory,
                                                               model_type,
                                                               noise_type,
                                                               ".tiff"))


            number_of_samples = len(x_files)
            print("number of samples {}".format(number_of_samples))
            # read and merge to one csv
            for growth_curve_image, derivative_image in zip(x_files,
                                                       derivative_files):

                snp1 = growth_curve_image.split(r"\\")[-1].split(".")[-2].split("_")[-6]
                snp2 = growth_curve_image.split(r"\\")[-1].split(".")[-2].split("_")[-5]
                snp3 = growth_curve_image.split(r"\\")[-1].split(".")[-2].split("_")[-4]
                snp4 = growth_curve_image.split(r"\\")[-1].split(".")[-2].split("_")[-3]
                count_number = growth_curve_image.split(r"\\")[-1].split(".")[-2].split("_")[-2]
                print(snp1,snp2,snp3,snp4,count_number)
                # read growth curve (biomass)
                growth_curve_array = cv2.imread(growth_curve_image)
                # Convert the image to grayscale
                gray_growth_curve_array = cv2.cvtColor(growth_curve_array,
                                                       cv2.COLOR_BGR2GRAY)
                gray_growth_curve_array = cv2.normalize(
                    gray_growth_curve_array, None, 0, 1.0,
                    cv2.NORM_MINMAX,
                    dtype=cv2.CV_32F)

                # read and concat derivative
                derivative_array = cv2.imread(derivative_image)
                # Convert the image to grayscale
                gray_derivative_array = cv2.cvtColor(derivative_array,
                                                     cv2.COLOR_BGR2GRAY)
                gray_derivative_array = cv2.normalize(gray_derivative_array,
                                                      None, 0, 1.0,
                                                      cv2.NORM_MINMAX,
                                                      dtype=cv2.CV_32F)


                input_images = np.stack([gray_growth_curve_array,gray_derivative_array],axis=0)
                print(input_images.shape)
                # Add the image and its snplabel to the dataset arrays
                image_np_list.append([input_images])
                ## read and concat snps
                new_snp_df = pd.DataFrame({"snp1":snp1,"snp2":snp2,"snp3":snp3,"snp4":snp4},index=[0])  # ,columns=['snp1','snp2','snp3','snp4']

                snp_label = pd.concat([snp_label, new_snp_df],
                                      ignore_index=True)
            else:
                # read model type label
                model_type_label = pd.DataFrame(columns=["model_type"],
                                                index=range(number_of_samples))
                model_type_label = model_type_label.assign(model_type=label_dict[model_type])
                print(model_type_label)
                sde_model_type_label = pd.concat(
                    [sde_model_type_label, model_type_label],
                    ignore_index=True)
        else:

            data = torch.tensor(np.array(
                image_np_list).squeeze())  # shape (n_samples,n_features,height,width)
            print(snp_label)
            snp_label = torch.tensor(snp_label.values.astype(np.float32))  # (n_samples,4)
            print("data shape:{}".format(data.shape))
            print("snp label tensor")
            print(snp_label.shape)
            print(snp_label)
            # one hot encoding for model label
            print(sde_model_type_label)
            sde_model_type_label = torch.tensor(sde_model_type_label.values.astype(np.float32))
            labels = one_hot_encoding_for_multiclass_classification(sde_model_type_label)

            print("datashape before saving")
            print(data.shape)
            # save input as dill file
            with open("{}{}_Input_X".format(directory,noise_type), "wb") as dillfile:
                dill.dump(data, dillfile)
            #torch.save(data,"{}plot/{}_Input_X.pt".format(directory,noise_type))
            with open("{}{}_Input_snps_label".format(directory,noise_type), "wb") as dillfile:
                dill.dump(snp_label, dillfile) #save snp labels without onehotencoding
            with open("{}{}_Input_sde_label".format(directory, noise_type),
                      "wb") as dillfile:
                dill.dump(labels,
                          dillfile)  # save model labels after onehotencoding

def read_biomass_and_filter():
    """
    reformat the ELOPE biomass biomass data(image DHline) later use for logistic model fitting
    :return:
    """
    # read file
    image_DH_line = pd.read_csv(
        "biomass_remove_single_outlier.csv",header=0,index_col=0)
    print(image_DH_line)
    print(image_DH_line.columns)

    #image_DH_line['Day'] = image_DH_line['timePoint'].str.split(':').str[-1].astype(int)

    image_DH_line = image_DH_line[['genotype',"plantid",'Biomass_Estimated','Day']]

    image_DH_line =image_DH_line.sort_values(by=['Day'])
    #calculate average biomass based on gennotype
    genotype_average_data = image_DH_line.groupby(['genotype','Day'])['Biomass_Estimated'].mean()

    print(len(image_DH_line['plantid'].unique()))
    print(len(image_DH_line['genotype'].unique()))
    # Merge genotype average data with original data
    merged_data = pd.merge(image_DH_line, genotype_average_data, on=['genotype','Day'],
                           suffixes=('', '_mean'))
    merged_data = merged_data.drop(columns=['plantid','Biomass_Estimated'])
    merged_data = merged_data.drop_duplicates()

    # # Filter out any rows where the biomass is 0 or NaN
    filtered_data = merged_data.loc[(merged_data['Biomass_Estimated_mean'] != 0) & (
        ~merged_data['Biomass_Estimated_mean'].isna())]

    print(filtered_data)

    genotype_groups = filtered_data.groupby(['genotype'])
    reset_index_group = pd.DataFrame()
    # create a continious time sequences from 2021-02-05 to 2021-03-24
    time_step_df = pd.DataFrame(pd.date_range("02/05/2021", freq="D", periods=48),columns=['Day']).astype('string')
    print(time_step_df)
    # Shift the days to start from 0 save in column day
    for group in genotype_groups.groups:
        group_df = genotype_groups.get_group(group)
        gene= group_df["genotype"].unique()[0]
        group_df = group_df.sort_values(by=['Day'])
        end_date = group_df.iloc[-1,1]
        print(end_date)
        #merge with time sequences to make sure every time series has exactly same days
        group_df['Day'] = group_df['Day'].astype('string')
        group_df = group_df.merge(time_step_df,on= 'Day',how= 'right')
        group_df['genotype'] = gene
        #find end date index, to avoid the last day's Biomass after merge is NA
        end_index = group_df.index[group_df['Day'] == end_date].tolist()[0]
        print(end_index)
        group_df = group_df.iloc[int(end_index)-21:int(end_index),:]
        group_df = group_df.reset_index(0,drop=True)
        group_df['Day'] = group_df.index
        reset_index_group = pd.concat([reset_index_group,group_df], ignore_index=True)
    print(reset_index_group)

    reset_index_group.to_csv("biomass_average_based_on_genotype.csv")

def read_and_cpmbine_simulated_data_with_gene_effect(directory:str="data/simulated_data/simulated_with_different_gene_type/"):
    """
    read csv file, combine data for LSTM model
    :param directory: the directory of files
    :return:
    """

    models = ["irradiance", "logistic", "Allee", "Temperature"]
    noises = ['time_independent_noise_0.25', 'time_dependent_noise_0.2', 'biomass_dependent_noise_0.2','without_noise']
    for noise_type in noises:
        snp_label = pd.DataFrame()
        sde_model_type_label = pd.DataFrame()
        models_biomass_df = pd.DataFrame()
        derivative_df = pd.DataFrame()
        for model_type in models:

            x_files = glob.glob("{}simulated_X_data_{}_{}_*{}".format(directory,model_type,noise_type, ".csv"))
            derivative_files = glob.glob("{}simulated_derivative_data_{}_{}_*{}".format(directory,model_type,noise_type, ".csv"))
            y_files = glob.glob("{}simulated_label_data_{}_{}_*{}".format(directory,model_type,noise_type, ".csv"))
            print(x_files)

            one_model_snp = pd.DataFrame()
            one_model_type_label = pd.DataFrame()
            one_model_biomass = pd.DataFrame()
            one_model_derivative = pd.DataFrame()
            # read and merge to one csv
            for x_file,derivative_file,y_file in zip(x_files,derivative_files,y_files):

                snp1 = x_file.split(r"\\")[-1].split(".")[-2].split("_")[-4]
                snp2 = x_file.split(r"\\")[-1].split(".")[-2].split("_")[-3]
                snp3 = x_file.split(r"\\")[-1].split(".")[-2].split("_")[-2]
                snp4 = x_file.split(r"\\")[-1].split(".")[-2].split("_")[-1]

                # read and concat biomass
                df_new = pd.read_csv(x_file, header=0, index_col=0)
                models_biomass_df = pd.concat([models_biomass_df, df_new],axis=1,ignore_index=True) # concat at column
                one_model_biomass = pd.concat([one_model_biomass, df_new],
                                              axis=1,ignore_index=True)
                #read and concat snps
                number_of_samples = len(df_new.columns)
                new_snp_df = pd.DataFrame(index=list(range(number_of_samples))) #,columns=['snp1','snp2','snp3','snp4']


                new_snp_df = new_snp_df.assign(snp1=snp1)
                new_snp_df = new_snp_df.assign(snp2=snp2)
                new_snp_df = new_snp_df.assign(snp3=snp3)
                new_snp_df = new_snp_df.assign(snp4=snp4)

                one_model_snp = pd.concat([one_model_snp,new_snp_df],ignore_index=True) #label df are concat at row
                snp_label = pd.concat([snp_label, new_snp_df],ignore_index=True)

                #read and concat derivative
                df_derivative_new = pd.read_csv(derivative_file,header=0,index_col=0)
                derivative_df = pd.concat([derivative_df,df_derivative_new],ignore_index=True,axis=1)
                one_model_derivative = pd.concat([one_model_derivative, df_derivative_new],
                                          ignore_index=True,axis=1)

                #read model type label
                model_type_label = pd.read_csv(y_file, header=0, index_col=0)
                print(model_type_label)
                sde_model_type_label = pd.concat([sde_model_type_label,model_type_label],ignore_index=True,axis=1)
                one_model_type_label = pd.concat([one_model_type_label,model_type_label],ignore_index=True,axis=1)
            else:
                #print(models_biomass_df)
                #save differnt model inputX and label sepreately
                #save to csv
                one_model_biomass.to_csv(str(directory+'reformat_data/simulated_X_data_' + model_type + "_"+noise_type+".csv"))
                one_model_derivative.to_csv(str(directory+'reformat_data/simulated_derivative_data_' + model_type + "_"+noise_type+".csv"))
                one_model_type_label.to_csv(str(directory+'reformat_data/simulated_label_data_' + model_type + "_"+noise_type+".csv"))
                one_model_snp.to_csv(str(directory+'reformat_data/simulated_snp_data_' + model_type + "_"+ noise_type+".csv"))
        else:
            print(models_biomass_df)
            print(sde_model_type_label)
            print(snp_label)
            print()
            models_biomass_df.to_csv(
                "{}reformat_data/{}_Input_X.csv".format(directory,noise_type))
            derivative_df.to_csv("{}reformat_data/{}_Input_derivative.csv".format(directory,noise_type))
            sde_model_type_label.to_csv("{}reformat_data/{}_Input_label.csv".format(directory,noise_type))
            snp_label.to_csv("{}reformat_data/{}_Input_snp.csv".format(directory,noise_type))

def anova_test_for_model_comparision():
    #This function is to perform anova test and t test with ten times repeat for
    # on the best model for SNP prediction and growth model prediction
    #and plot the average result accuracy and copen's kappa score with sd
    result_df = pd.read_csv("CNN_10_repeates_result_4models.csv",header=0,index_col=0)
    print(result_df.columns)
    groups = result_df.groupby("noise_type")
    #groups = result_df.groupby("snp")
    for group in groups.groups:
        print(group)
        group_df = groups.get_group(group)
        print(group_df.mean().round(2))
        # t test between result and random
        from scipy.stats import ttest_1samp

        null_hypothesis = 0.5  # dependes on number of classes
        t_statistic, p_value = ttest_1samp(group_df["test_accuracy"], null_hypothesis,alternative='greater')
        print(p_value,t_statistic)
    result_df = result_df[["noise_type",'test_accuracy', 'cohenkappa_score']]
    #result_df = result_df[["snp", 'test_accuracy', 'cohenkappa_score']]
    # Convert the data to long format
    df_melt = pd.melt(result_df, id_vars=['noise_type'], value_vars=['test_accuracy', 'cohenkappa_score'],
                      var_name='Metric', value_name='Score')
    # df_melt = pd.melt(result_df, id_vars=['snp'], value_vars=['test_accuracy', 'cohenkappa_score'],
    #                   var_name='Metric', value_name='Score')
    df_melt.drop_duplicates(inplace=True)
    print(df_melt)
    # # Perform two-way ANOVA
    # rm = AnovaRM(df_melt, 'Score', 'noise_type', within=['Metric'], aggregate_func='mean')
    # res = rm.fit()

    # Print the ANOVA table
    # print(res.anova_table)

    # Plot the results
    sns.pointplot(x='noise_type', y='Score', hue='Metric', data=df_melt, ci='sd',
                  capsize=.2, join=False)
    # sns.pointplot(x='snp', y='Score', hue='Metric', data=df_melt, ci='sd',
    #               capsize=.2, join=False)
    plt.ylim(0.0, 0.8)
    #plt.xticks([0,1,2,3],["A","B","C","D"])
    plt.title("CNN growth model classification")
    plt.show()


def main():
    result_df_lstm = pd.read_csv("cnn_10_repeates_result_4models.csv", header=0,
                            index_col=0)
    result_df_lstm["network_type"] = "CNN"
    result_df_cnn = pd.read_csv("lstm_30_repeates_result_4models_without_derivative.csv", header=0,
                            index_col=0)
    result_df_cnn["network_type"] = "LSTM"
    data_df = pd.concat([result_df_lstm,result_df_cnn],ignore_index=True)

    groups = data_df.groupby("noise_type")
    # groups = result_df.groupby("snp")
    for group in groups.groups:
        print(group)
        group_df = groups.get_group(group)
        print(group_df.mean().round(2))
    result_df = data_df[["noise_type", 'test_accuracy', "network_type"]]
    order = result_df.groupby('noise_type')['test_accuracy'].mean().sort_values(
        ascending=False).index
    print(result_df)
    sns.set(font_scale=1.3)
    sns.pointplot(x='noise_type', y='test_accuracy', hue="network_type", data=result_df, ci='sd',
                  capsize=.1, join=False,order=order)

    plt.ylim(0.0, 1.0)

    plt.title("Growth model classification")
    plt.show()
    '''
    from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
    result_df = pd.read_csv("_test_snps_CNN4.csv")
    y= result_df[["snp_4"]]
    y_predict = result_df[["predict_labelAdam_lr0.01_epoch500_batch32kernel3stride12noisewithout_noise"]]
    cm = confusion_matrix(y, y_predict)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=["0",
                                                            "2",
                                                            ]).plot()
    plt.title(
        "SNP_temperature_tolerance confusion matrix ")

    plt.show()
    '''
    #anova_test_for_model_comparision()
    #read_biomass_and_filter()
    #read_and_cpmbine_simulated_data_with_gene_effect()
    #read_image_and_reformat_for_snps_prediction()
    # save_combined_feature_tensors_as_dill_files( #save input for LSTM model (growth curve and derivative)
    #     folder="data/simulated_data/simulated_from_elope_data_120_no_gene_effect/")
    #read_image_and_reformat(folder="data/simulated_data/simulated_from_elope_data_120_no_gene_effect/")
    #save_combined_feature_tensors_as_dill_files()
    #unittest.main()
    #read_reformat("./data/")
    # for name in ["Merredin","Narrabri","Yanco"]:
    #     file_name="data/{}_data.csv".format(name)
    #     read_and_reformat_for_clustering(file_name)

if __name__ == '__main__':
    main()
    