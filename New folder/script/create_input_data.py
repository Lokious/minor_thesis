import glob
import os
import torch
import pandas as pd
import numpy as np
import dill


def merge_snp_with_genotype(selected_snps:pd.DataFrame,genotypes:pd.DataFrame):
    """
    only keep the SNPs which were choosen

    :param selected_snps: contains  selected snps and it position
    :param genotypes: contains all genotypes and snps type:0,1,2
    :return: pd.Dataframe
    """

    selected_snps_list = list(selected_snps.index)
    selcted_genotype_snps_df = genotypes[selected_snps_list]
    print(selcted_genotype_snps_df)
    return selcted_genotype_snps_df


def merge_SNP_with_traits_df(traits_df,selected_snp_df,input_dir):
    """
    This function is to make sure the order of label is corresponding to the plantid/potid
    if we take the average of plants use the genotype instead, this function is not necessary

    :param traits_df:
    :param selected_snp_df:
    :param input_dir:
    :return:
    """
    raise RuntimeError("the function is not use anymore")
    selected_snp_df['genotype_name'] = list(selected_snp_df.index)
    selected_snp_df = selected_snp_df.reset_index(drop=True)
    #keep genotype_name and related plantid
    try:
        traits_df = traits_df[['genotype_name',"plantid"]]
        print("use plantid")
    except:
        traits_df = traits_df[['genotype', "plotId"]]
        print("use plotId")
        traits_df=traits_df.rename(columns={'genotype':'genotype_name'})



    # merge based on genotype:
    pd_new = traits_df.merge(selected_snp_df, on="genotype_name")

    # print(set(list1)-set(list2))
    pd_new = pd_new.drop_duplicates()
    isExist = os.path.exists("{}/".format(input_dir))
    if not isExist:
        os.makedirs("{}/".format(input_dir))
    print(pd_new)
    pd_new.to_csv("{}/merged_trait_snps.csv".format(input_dir))

    return pd_new

def data_prepare(simulated_dataset,label_df,snp,save_directory="../data/input_data"):
    """
    logistic early growth traits data
    :return: input_x,input_y,n_features
    """
    def create_tensor_dataset(dfs):
        #(41,418,3) if use three trait raw data
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

    tensor_dataset, n_features = create_tensor_dataset(simulated_dataset)
    # creating tensor from targets_df
    tensor_lable = torch.tensor(label_df[snp].values.astype('float'))
    tensor_lable = torch.unsqueeze(tensor_lable,1)
    #print(tensor_lable)
    # the number represent the number in old order, and after that will place it
    # as the new order for example: permute(2,0,1)
    # (a,b,c) -> (c,a,b)
    tensor_dataset = torch.permute(tensor_dataset, (2, 1, 0)) # (n_seq,seq_len,n_features) 1200,45,3

    #creat folder to save input files
    isExist = os.path.exists("{}/".format(save_directory))
    if not isExist:
        os.makedirs("{}/".format(save_directory))
    with open("{}/input_X".format(save_directory),"wb") as dillfile1:
        dill.dump(tensor_dataset,dillfile1)
    with open("{}/input_Y_{}".format(save_directory,snp),"wb") as dillfile2:
        dill.dump(tensor_lable,dillfile2)

    return tensor_dataset,tensor_lable,n_features

def create_input_data(snp_df: str = "../data/chosen_snps_map.csv",
                      geno_df: str = "../data/data_geno_genotype",
                      Xs: list = [],
                      save_directory: str = "../data/input_data",
                      ):

    # read file
    snp_df = pd.read_csv(snp_df, header=0, index_col=0)

    snp_list = list(snp_df.index)
    print(snp_list)
    genotype_df = dill.load(open(geno_df, "rb"))
    genotype_df = genotype_df.replace({2: 1})
    selected_snp_df = merge_snp_with_genotype(snp_df, genotype_df)
    # print(selected_snp_df)
    drop_index = []
    # print(len(selected_snp_df.columns))
    for col in selected_snp_df.columns:

        if len(selected_snp_df[col].unique()) == 1: #if only have one type of snp
            drop_index.append(col)
            snp_list.remove(col)
    selected_snp_df = selected_snp_df.drop(columns=drop_index)
    # print(len(selected_snp_df.columns))

    # read get the genotype_name order of inputX
    traits_df = Xs[0]
    traits_df_order_genotype = pd.DataFrame()
    traits_df_order_genotype["genotype_name"] = traits_df.columns

    selected_snp_df['genotype_name'] = list(selected_snp_df.index)
    selected_snp_df = selected_snp_df.reset_index(drop=True)

    merge_traits_snps = traits_df_order_genotype.merge(selected_snp_df,on="genotype_name")


    #merge_traits_snps = merge_SNP_with_traits_df(traits_df, selected_snp_df,save_directory)


    for snp in snp_list:
        data_prepare(simulated_dataset=Xs, label_df=merge_traits_snps, snp=snp,save_directory=save_directory)

def keep_overlap_plant_for_input_data(file_directory:str="../data/spline_extract_features/",file_name_end_with:str="*_reformat.csv"):

    files = glob.glob("{}{}".format(file_directory,file_name_end_with))
    print(files)
    df_list=[]
    for file in files:
        df_new = pd.read_csv(file,header=0,index_col=0)
        df_list.append(df_new)

    columns_name_list = []
    for df_new in df_list:
        #find overlap columns
        columns_name_list.append(list(df_new.columns))

    overlap_plant_id = set.intersection(*[set(x) for x in columns_name_list])
    overlap_dfs_list = [] # list of pd only keep overlap potId/plantid
    for df in df_list:
        df_remove_no_overlap = df[list(overlap_plant_id)]
        overlap_dfs_list.append(df_remove_no_overlap)

    #files = glob.glob("{}{}".format(file_directory, "*predict_withp_spline.csv"))


    # # save traits files (before reformat) with only overlap plantid,
    # # use for create input_Y
    # for file in files:
    #     print(file)
    #     df = pd.read_csv(file,header=0,index_col=0)
    #     df = df[df["plotId"].isin(overlap_plant_id)]
    #     file_name=file.split("\\")[1]
    #     save_file_name = file_directory + file_name.split("_")[0]+"_overlap_plotId_average.csv"
    #     df.to_csv(save_file_name)
    #
    return overlap_dfs_list

def remove_plant_with_larger_than_five_dayas_missing(df_list):
    raise RuntimeError("!")
    # if not, 35 days left
    overlap_dfs = []
    for df_with_no_overlap in df_list:

        print(len(df_with_no_overlap.columns))
        plots = list(df_with_no_overlap.columns)
        missing_days = {}
        for i in range(46):
            missing_days[i] = 0

        for col in plots:
            num_missing_day = df_with_no_overlap[col].isna().sum()
            missing_days[num_missing_day] += 1
            # if one plant have larger than 5 days missing , we will remove it
            # if df_with_no_overlap[col].isna().sum() > 5:
            #     print(col)
            #     df_with_no_overlap = df_with_no_overlap.drop(columns=[col])
        # # remove the start and end day wich includes nan
        print(df_with_no_overlap)
        df_with_no_overlap = df_with_no_overlap.dropna()
        print(df_with_no_overlap)
        overlap_dfs.append(df_with_no_overlap)

        import matplotlib.pyplot as plt
        missing_days = pd.DataFrame(missing_days,index=[0])
        print(missing_days)
        plt.plot(missing_days.T)
        plt.show()
        return overlap_dfs

def calculate_average_for_every_genotype(df_list,plant_genotype_map_df_dir):

    plant_genotype_map_df = pd.read_csv(plant_genotype_map_df_dir,header=0,index_col=0)[["genotype_name","plantid","XY"]]
    print(plant_genotype_map_df)
    map_dictionary = {}
    group_object = plant_genotype_map_df.groupby("genotype_name")
    for group_name in group_object.groups:
        group_df = group_object.get_group(group_name)
        map_dictionary[group_name]= list(group_df["XY"].unique())
    print(map_dictionary)
    genotype_list = map_dictionary.keys()
    average_df_list=[]
    for df in df_list:

        average_df = pd.DataFrame()
        for genotype in genotype_list:
            pot_id_list = map_dictionary[genotype]
            if not df[df.columns[df.columns.isin(pot_id_list)]].empty:
                average_df[genotype] = (df[df.columns[df.columns.isin(pot_id_list)]]).copy().mean(axis=1) # calculate average for pot with the same genotype
                #average_df = pd.concat([average_df,df[df.columns[df.columns.isin(pot_id_list)]].copy().mean(axis=1)],axis=1)
        else:
            # dropna will lead to not same length sequences
            # average_df = average_df.dropna(how='any')
            print(average_df)
            average_df_list.append(average_df)
    return average_df_list

def main():
    # # read reformat df
    # new_df_LA = pd.read_csv("../data/df_LA.csv", header=0, index_col=0)
    # print(new_df_LA)
    # new_df_Height = pd.read_csv("../data/df_Height.csv", header=0, index_col=0)
    # new_df_Biomass = pd.read_csv("../data/df_Biomass.csv", header=0,
    #                              index_col=0)
    # Xs = [new_df_LA, new_df_Height, new_df_Biomass]
    #
    #
    Xs = keep_overlap_plant_for_input_data()
    Xs = calculate_average_for_every_genotype(Xs,"../data/image_DHline_data.csv")
    #print(Xs)

    create_input_data(Xs=Xs,save_directory="../data/input_data/spline_predict_input_average")

if __name__ == '__main__':
    main()
