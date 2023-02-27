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


def merge_SNP_with_traits_df(traits_df,selected_snp_df):

    selected_snp_df['genotype_name'] = list(selected_snp_df.index)
    selected_snp_df = selected_snp_df.reset_index(drop=True)
    traits_df = traits_df[['genotype_name',"plantid"]]
    print(len(traits_df["genotype_name"].unique()))
    list1 =list(traits_df["genotype_name"].unique())
    # merge based on genotype:
    pd_new = traits_df.merge(selected_snp_df, on="genotype_name")
    list2 = list(pd_new["genotype_name"].unique())
    print(set(list1)-set(list2))
    pd_new = pd_new.drop_duplicates()
    print(pd_new)
    pd_new.to_csv("../data/merged_trait_snps.csv")

    return pd_new

def data_prepare(simulated_dataset,label_df,snp,save_directory="../data/input_data"):
    """
    logistic early growth traits data
    :return: input_x,input_y,n_features
    """
    def create_tensor_dataset(dfs):
        #(41,418,3)
        datasets = []
        for df in dfs:
            sequences = df.astype(np.float32).to_numpy().tolist()
            dataset = torch.stack([torch.tensor(s).unsqueeze(1).float() for s in sequences])
            datasets.append(dataset)

        #remove the last dimention
        tensor_dataset = torch.squeeze(torch.stack(datasets,dim=0))
        # print("shape of created dataset:")
        # print(tensor_dataset.shape)

        n_features,seq_len,n_seq= tensor_dataset.shape

        return tensor_dataset,n_features

    tensor_dataset,n_features = create_tensor_dataset(simulated_dataset)
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

def create_input_data(snp_df:str="../data/chosen_snps_map.csv",geno_df:str="../data/data_geno_genotype",Xs:list=[],save_directory:str="../data/input_data"):
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

        if len(selected_snp_df[col].unique()) == 1:
            drop_index.append(col)
            snp_list.remove(col)
    selected_snp_df = selected_snp_df.drop(columns=drop_index)
    # print(len(selected_snp_df.columns))

    # read traits df without spatial correction
    traits_df = pd.read_csv(
        "../data/image_DHline_data_after_average_based_on_day.csv", header=0,
        index_col=0)

    merge_traits_snps = merge_SNP_with_traits_df(traits_df, selected_snp_df)


    for snp in snp_list:
        data_prepare(Xs, merge_traits_snps, snp,save_directory=save_directory)

def keep_overlap_plant_for_input_data(file_directory:str="../data/spline_extract_features/",file_name_end_with:str="*_reformat.csv"):

    files = glob.glob("{}{}".format(file_directory,file_name_end_with))
    print(files)
    df_list=[]
    overlap_df = []
    columns_name_list = []
    # print(columns_name_list)
    for file in files:
        df_new = pd.read_csv(file,header=0,index_col=0)
        # print(df_new)
        df_list.append(df_new)
        columns_name_list.append(list(df_new.columns))
    print(columns_name_list)
    overlap_plant_id = set.intersection(*[set(x) for x in columns_name_list])
    print(overlap_plant_id)
    print(len(overlap_plant_id))
    for df in df_list:
        df_remove_no_overlap = df[list(overlap_plant_id)]
        overlap_df.append(df_remove_no_overlap)
    return overlap_df

def main():
    # # read reformat df
    new_df_LA = pd.read_csv("../data/df_LA.csv", header=0, index_col=0)
    print(new_df_LA)
    # new_df_Height = pd.read_csv("../data/df_Height.csv", header=0, index_col=0)
    # new_df_Biomass = pd.read_csv("../data/df_Biomass.csv", header=0,
    #                              index_col=0)
    # Xs = [new_df_LA, new_df_Height, new_df_Biomass]
    #

    Xs = keep_overlap_plant_for_input_data()


    create_input_data(Xs=Xs,save_directory="../data/input_data/spline_predict_input")

if __name__ == '__main__':
    main()
