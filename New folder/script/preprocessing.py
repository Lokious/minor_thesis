"""
This code is to preprocess the phenotype and genotype data

save the dataframe using dill; csv file tooks more time and memory while reading
"""
import copy
import random
import dill
#import pyreadr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from scipy.stats import chi2_contingency
import math
import unittest

import glob
def read_rdata_to_csv():


    # read field data
    data_field = pyreadr.read_r(
        '../../elope-main/ELOPE_raw_data/data_field/field_pheno_filtered.RData')
    print(data_field['field_lines'])
    DH_field_data = data_field['field_lines']
    print(DH_field_data.columns)
    hybrids_field_data = data_field['field_hybrids']
    # ## drop NA
    # DH_field_data = DH_field_data.dropna()
    # hybrids_field_data = hybrids_field_data.dropna()

    ## save as csv file (for looking in excel) and save object (for using in code)
    with open("../data/field_DHline_data", "wb") as dill_file:
        dill.dump(DH_field_data, dill_file)
    with open("../data/field_hybrids_data", "wb") as dill_file:
        dill.dump(hybrids_field_data, dill_file)
    # DH_field_data.to_csv("../data/field_DHline_data.csv")
    # hybrids_field_data.to_csv("../data/field_hybrids_data.csv")

    # read platform image data
    data_platform = pyreadr.read_r(
        "../../elope-main/ELOPE_raw_data/data_platform/platform_image_filtered.RData")

    DH_platform_data = data_platform['image_lines']
    print(DH_platform_data.columns)
    hybrids_platform_data = data_platform['image_hybrids']
    # # drop NA
    # DH_platform_data = DH_platform_data.dropna()
    # hybrids_platform_data = hybrids_platform_data.dropna()

    ## save as csv file (for looking in excel) and save object (for using in code)
    # DH_platform_data.to_csv("../data/image_DHline_data.csv")
    # hybrids_platform_data.to_csv("../data/image_hybrids_data.csv")
    with open("../data/image_DHline_data", "wb") as dill_file:
        dill.dump(DH_platform_data, dill_file)
    with open("../data/image_hybrids_data", "wb") as dill_file:
        dill.dump(hybrids_platform_data, dill_file)

    data_manual_platform = pyreadr.read_r("../../elope-main/ELOPE_raw_data/data_platform/platform_pheno_filtered.RData")
    data_manual_platform_DH = data_manual_platform["pheno_lines"]
    data_manual_platform_hybrids = data_manual_platform["pheno_hybrids"]
    print(data_manual_platform_DH)
    ## save as csv file (for looking in excel) and save object (for using in code)
    data_manual_platform_DH.to_csv("../data/data_manual_platform_DH.csv")
    data_manual_platform_hybrids.to_csv("../data/data_manual_platform_hybridscsv")
    with open("../data/data_manual_platform_DH", "wb") as dill_file:
        dill.dump(data_manual_platform_DH, dill_file)
    with open("../data/data_manual_platform_hybrids", "wb") as dill_file:
        dill.dump(data_manual_platform_hybrids, dill_file)

    # read genotype file
    data_geno = pyreadr.read_r(
        "../../elope-main/ELOPE_raw_data/data_genotypic/geno_filtered.RData")
    #print(data_geno)
    data_geno_genotype = data_geno['geno']
    data_geno_map = data_geno['map']
    # # drop NA
    # data_geno_genotype = data_geno_genotype.dropna()
    # data_geno_map = data_geno_map.dropna()
    ## save as csv file (for looking in excel) and save object (for using in code)
    # data_geno_genotype.to_csv("../data/data_geno_genotype.csv")
    # data_geno_map.to_csv("../data/data_geno_map.csv")
    with open("../data/data_geno_genotype", "wb") as dill_file:
        dill.dump(data_geno_genotype, dill_file)
    with open("../data/data_geno_map", "wb") as dill_file:
        dill.dump(data_geno_map, dill_file)

    return DH_field_data,hybrids_field_data,DH_platform_data,hybrids_platform_data,data_geno_genotype,data_geno_map


def log_transform(raw_df:pd.DataFrame,columns:list)->pd.DataFrame:
    """
    Still keep the NA after log(x+1) transform
    :param raw_df:
    :param columns:
    :return:
    """

    print(raw_df)
    #log transform (use log(x+1) to deal with zero value)for LA, height and biomass
    new_columns = [(x+"_log_transformed") for x in columns]
    raw_df[new_columns]=raw_df[columns].apply(lambda x:np.log(x+1))
    print(raw_df[new_columns])
    return raw_df


def check_genotype(dataframe:pd.DataFrame,check_geno="CH"):
    """
    Return genotype name for Check genotypes

    :param dataframe:
    :param check_geno:
    :return:
    """

    indexes=dataframe[dataframe['m'] == check_geno].index
    # m is related to genotype_name in platform data
    check_genotype = list(set(dataframe.loc[indexes,'InbredCode']))
    print(check_genotype)
    return check_genotype


def remove_check_genotype(dataframe:pd.DataFrame,check_geno:list):

    print(dataframe)
    length1= len(dataframe.index)
    dataframe.drop(dataframe[dataframe.genotype_name.isin(check_geno)].index, inplace=True)
    length2 = len(dataframe.index)
    print("remove {} rows as check genotype".format(length1-length2))
    return dataframe

def calculate_average_traits_value_by_day(trait_df:pd.DataFrame)->pd.DataFrame:

    # only keep useful columns
    trait_df = trait_df[
        ["plantid","LA_Estimated", "Height_Estimated", "Biomass_Estimated",
         "genotype_name", "DAS","Line","Position","XY","datetime","Day","Rep"]]
    non_numical_column=trait_df[["plantid","Day","genotype_name", "DAS","Line","Position","XY","Rep"]]
    print(non_numical_column)
    non_numical_column = non_numical_column.drop_duplicates()
    print(non_numical_column)
    # every plant should be measured once per day, for those measured multiple times, calculate mean
    after_average = trait_df.groupby(["plantid","Day"])["LA_Estimated", "Height_Estimated", "Biomass_Estimated"].mean().reset_index()
    print(after_average)
    useful_column=after_average.merge(non_numical_column,on=["plantid","Day"])
    print(useful_column)
    #useful_column.to_csv("../data/image_DHline_data_after_average_based_on_day.csv")

    return useful_column

def get_colors(num_colors):
    import colorsys
    colors = []
    random.seed(0)
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i / 360.
        lightness = (50 + np.random.rand() * 10) / 100.
        saturation = (90 + np.random.rand() * 10) / 100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

def remove_no_effect_SNPs(geno_df:pd.DataFrame,imbalanced_control=False)->int:

    """
    remove thoses SNPs which is the same among all 955 genotypes

    :param trait_df:
    :param geno_df:
    :return:
    """
    print(geno_df.shape)
    snp_list = geno_df.columns
    percentage_dictionary = {}
    geno_df = geno_df.astype("int16")
    geno_df["genotype_name"] = geno_df.index
    geno_df.reset_index(drop=True)
    for i,snp in enumerate(snp_list):
        if len(geno_df[snp].unique()) == 1:
            # print(geno_df[snp].unique())
            # print("drop")
            geno_df.drop(columns=[snp],inplace=True)
        else:
            percentage = geno_df[snp].value_counts(normalize=True)
            percentage_dictionary[snp] = percentage
            print(percentage)
            if imbalanced_control:
                if percentage<0.4 or percentage>0.6:
                    geno_df.drop(columns=[snp], inplace=True)
    print(geno_df.shape)
    # print(percentage_dictionary)
    # print(len(percentage_dictionary.keys()))


    #save the df after remove snps which is the same for all
    # with open ("../data/remove_no_effect_snps","wb") as dillfile:
    #     dill.dump(geno_df,dillfile)

    return 0

def select_SNPs_based_on_distance(map_gene_df):
    """
    The average physical to genetic ratio in the maize genome is 182 kb per
    cM with great variation, ranging from >1.8 Mb/cM in centromeric regions
    to <10 kb in telomeric region.
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1934398/
    :return:
    """
    #I will select one SNP per 1.82Mb
    coverntion_constant = 1820000

    #group by Chromesome
    groups_objects = map_gene_df.groupby("chr")
    chosen_snp_df = pd.DataFrame()
    for group in groups_objects.groups:
        chr_group = groups_objects.get_group(group)
        print(chr_group)
        chr = chr_group["chr"].unique()[0]
        print(chr)
        snp_list = list(chr_group.index)
        #print(snp_list)
        keep_snp_list = []
        while snp_list != []:
            snp_id = snp_list.pop(0)
            snp_pos = chr_group.loc[snp_id,"pos"]
            #print(snp_pos)
            if keep_snp_list == []:
                keep_snp_list.append(snp_id)
            else:
                #compare the distance between current position and next SNP's position
                current_pos = chr_group.loc[keep_snp_list[-1],"pos"]
                distance = snp_pos - current_pos
                if distance > coverntion_constant:
                    print(distance)
                    keep_snp_list.append(snp_id)
        print("keep {} snps on chr{}".format(len(keep_snp_list),chr))
        chosen_snp_df = pd.concat([chosen_snp_df,map_gene_df.loc[keep_snp_list,:]])
    else:
        print(chosen_snp_df)
        return chosen_snp_df


def calculate_SNPs_relationship(snps_df:pd.DataFrame) -> pd.DataFrame:


    # select snps based on correlation?
    ## can not save as a dataframe due to memory error
    corr_snp_df = {}
    snps = list(snps_df.columns)
    print(snps)
    i = 0
    while snps!=[]:
        print(i)
        snp1=snps.pop()
        corr_snp_df[snp1]={}
        for snp2 in snps:
            corss_tab = pd.crosstab(index=snps_df[snp1],columns=snps_df[snp2])
            print(corss_tab)
            correlation = chi2_contingency(corss_tab)
            #print(correlation)
            corr_snp_df[snp1][snp2] = correlation[:2]
            #print(corr_snp_df)
        else:
            i +=1
    #save correlation df
    with open("../data/chi2_contingency_result","wb") as dillfile:
        dill.dump(corr_snp_df,dillfile)
    # find where corr = 1 or -1
    # where_snps = corr_snp_df.where(corr_snp_df==(1 or -1))
    # print(where_snps)
    return corr_snp_df

def merge_traits_SNPs(trait_df:pd.DataFrame,geno_df:pd.DataFrame)->pd.DataFrame:
    trait_df=trait_df[["LA_Estimated","Height_Estimated","Biomass_Estimated","genotype_name","DAS"]]
    print(trait_df.shape)
    merged_df = trait_df.merge(geno_df, on="genotype_name")
    print(merged_df)
    # for snp in snp_list:
    #     single_snp_df = geno_df[[snp,"genotype_name"]]
    #     merged_df = trait_df.merge(single_snp_df,on="genotype_name")
    #     print(merged_df.columns)
    #     print(merged_df)
    #     with open("../data/single_snp_merge_trait/{}".format(snp),"wb") as dillfile:
    #         dill.dump(merged_df,dillfile)
    return merged_df

def plot_raw_data(DH_platform_data):

    groups_object = DH_platform_data.groupby('plantid')
    group_number = len(groups_object.groups)
    color_list=get_colors(group_number)
    for item in groups_object.groups:
        #group based on plant, every plant related to one line

        #print(item)
        plant = groups_object.get_group(item)
        print(plant[['DAS',"plantid"]])
        # drop duplicate day( ned to to change to average ..)
        # plant = plant.drop_duplicates(subset=['DAS'])
        #print(plant)
        for day in plant.index:
            # plant_id = plant.loc[day, 'plantid']
            # pot_id = plant.loc[day, 'Pot']
            #check pot id equals plant id
            assert True
        else:
            #it has 4 different harvest day
            # print(plant['DAS'])
            # print(plant['LA_Estimated'])
            plt.plot(plant["DAS"],plant['Height_Estimated_log_transformed'],label=plant.loc[day, 'plantid'],linewidth=0.2)
            plt.axvline(x=plant.loc[list(plant["DAS"].index)[-1],"DAS"], linestyle = '-',linewidth=1)

            #plt.legend()
        # if item==20:
        #     break
    plt.ylabel("Height_Estimated_log_transformed")
    plt.xlabel("Days")
    #plt.savefig("Biomass_Estimated.png")
    plt.show()


def missing_value_count(raw_df: str = "../data/image_DHline_data_after_average_based_on_day_log.csv"):

    df = pd.read_csv(raw_df,header=0,index_col=0)
    days= list(set(df["DAS"].unique()))
    print(days)
    na_count = df.isna().sum().sum()
    print(na_count)
    number_of_values = len(df.index)*3
    # print the number of missing values
    print("values number: {}".format(number_of_values))
    # print the proportion of missing values
    print("{}".format((na_count/number_of_values)))

    # check if the number of missing vaules id randomly present in the whole dataset
    ## based on Days after sowing
    df_day_count = pd.DataFrame(index=days,columns=["count"])
    group_object = df.groupby("DAS")
    for item in group_object.groups:

        group_df = group_object.get_group(item)
        day=group_df["DAS"].unique()
        print(day)
        daily_na_count = group_df.isna().sum().sum()
        df_day_count.loc[day,"count"] = daily_na_count
    print(df_day_count)
    plt.plot(df_day_count)
    plt.show()

    ## based on genotypes
    genotypes= list(set(df["genotype_name"].unique()))
    df_genotype_count = pd.DataFrame(index=genotypes, columns=["count"])
    group_object = df.groupby("genotype_name")
    for item in group_object.groups:
        group_df = group_object.get_group(item)
        geneotype = group_df["genotype_name"].unique()
        #print(geneotype)
        daily_na_count = group_df.isna().sum().sum()
        df_genotype_count.loc[geneotype, "count"] = daily_na_count
    print(df_day_count)
    plt.plot(df_genotype_count)
    plt.show()

    #check if missing value count is randomly distribute
    import scipy.stats as stats
    import pylab
    x = df_genotype_count["count"].astype(float).to_numpy()

    k2, p_value = stats.normaltest(x)
    stats.probplot(x, dist="norm", plot=pylab)
    pylab.show()
    # tau, p_value = stats.kendalltau(["a","b","G","H"], [1,2,3,4])
    print(k2,p_value)
    x = df_day_count["count"].astype(float).to_numpy()

    k2, p_value = stats.normaltest(x)
    print(k2, p_value)

def generate_a_test_set_by_duplicate_the_data_from_several_genotypes_and_add_noise(raw_data):

    n=3
    repetition = 3
    genotype_list = raw_data["genotype_name"]
    simulated_data = pd.DataFrame()
    genotype_list = random.choices(genotype_list, k=n)
    print(genotype_list)
    for i in genotype_list:

        select_df = raw_data.loc[raw_data["genotype_name"]==i]
        #select_df = select_df.drop_duplicates(subset=["DAS","genotype_name"])
        simulated_data = pd.concat([simulated_data, select_df],ignore_index=True)
        print(simulated_data)
        for j in range(repetition):

            rep_select_df = copy.deepcopy(select_df)

            #print(rep_select_df.columns)
            #just to make sure assign differnt plantid
            rep_select_df["plantid"] = rep_select_df["plantid"]+(2000*(j+1))
            print(rep_select_df["plantid"])

            # #shift at time point
            # shift_days = random.randint(0,6)
            # rep_select_df["DAS"]= rep_select_df["DAS"]+shift_days
            #add noise
            # height_noise = [random.random() for x in
            #                 rep_select_df["Height_Estimated_log_transformed"]]
            # LA_noise = [random.random() for x in
            #             rep_select_df["LA_Estimated_log_transformed"]]
            # biomass_noise = [random.random() for x in rep_select_df[
            #     "Biomass_Estimated_log_transformed"]]
            height_noise=0
            LA_noise = 0
            biomass_noise=0
            rep_select_df["LA_Estimated_log_transformed"] = rep_select_df["LA_Estimated_log_transformed"] + LA_noise
            rep_select_df["Height_Estimated_log_transformed"] = rep_select_df[
                                                                "Height_Estimated_log_transformed"] + height_noise
            rep_select_df["Biomass_Estimated_log_transformed"] = rep_select_df[
                                                                "Biomass_Estimated_log_transformed"] + biomass_noise
            simulated_data = pd.concat([simulated_data, rep_select_df],ignore_index=True)
    else:
        print(len(genotype_list),j)
        print(simulated_data)
        #simulated_data.to_csv("../data/simulated_data_6_genotype_4_rep_shift_based_on_das.csv")
        #add noise data
        simulated_data.to_csv("../data/simulated_data_6_genotype_4_rep_nonoise.csv")

def pre_process_for_spline_extract_features(file):

    platform_data = pd.read_csv(file,header=0,index_col=0)

    print(platform_data)
    #take one value per day

    platform_data = platform_data[platform_data["timeNumber"].isin(list(range(46)))]
    platform_data["timePoint"] = platform_data["timeNumber"]
    # platform_data = platform_data[platform_data["timePoint"].str.endswith("00:00:00")]
    # platform_data["timePoint"] = platform_data["timePoint"].apply(lambda x: pd.Series(str(x).split(" ")[0]))
    print(platform_data)
    #platform_data = platform_data.fillna(0.0)
    # print(len(platform_data["genotype_name"].unique()))

    platform_data = platform_data[platform_data['genotype'].str.startswith('DH')]
    #platform_data.to_csv("../data/image_DHline_data_after_average_based_on_day.csv")
    groups_object = platform_data.groupby('plotId')
    group_number = len(groups_object.groups)
    print("group_num:{}".format(group_number))
    length = 0
    time_list = []
    plantid_list = []
    for item in groups_object.groups:

        # group based on plant, every plant related to one line
        plant_df = groups_object.get_group(item)

        plant_df.set_index("timePoint",inplace=True)
        #print(plant_df)
        time_length = len(plant_df.index)

        if time_length > length:
            length = time_length
            time_list = list(plant_df.index)
            #print("time steps:{}".format(time_length))
        #time_series_df_list.append(plant_df)
        plantid_list.append(plant_df["plotId"].unique()[0])
    #print(length)

    new_df_predict = pd.DataFrame(index=time_list,columns=plantid_list)
    new_df_deriv = pd.DataFrame(index=time_list, columns=plantid_list)
    new_df_deriv2 = pd.DataFrame(index=time_list, columns=plantid_list)
    platform_data.set_index(['plotId','timePoint'],inplace=True)
    for i in time_list:
        for column in plantid_list:
            try:
                #print(platform_data.loc[(column,i)])
                new_df_predict.loc[i,column] = platform_data.loc[(column,i),"pred.value"]
            except:
                new_df_predict.loc[i, column] = np.nan
            try:
                #print(platform_data.loc[(column,i)])
                new_df_deriv.loc[i,column] = platform_data.loc[(column,i),"deriv"]
            except:
                new_df_deriv.loc[i, column] = np.nan
            try:
                #print(platform_data.loc[(column,i)])
                new_df_deriv2.loc[i,column] = platform_data.loc[(column,i),"deriv2"]
            except:
                new_df_deriv2.loc[i, column] = np.nan


    print(new_df_predict)
    return new_df_predict,new_df_deriv,new_df_deriv2
def choose_snps():

    data_geno_genotype=dill.load(open("../data/data_geno_genotype","rb"))# need large memory if read csv,use dill load previous saved file instead
    remove_no_effect_SNPs(data_geno_genotype,imbalanced_control=True)
    #selcet based on distance
    #data_geno_map= dill.load(open("../data/remove_no_effect_snps", "rb"))
    #print(data_geno_genotype)
    #chonsen_snps_mapdf = select_SNPs_based_on_distance(data_geno_map)
    # chonsen_snps_mapdf.to_csv("../data/chosen_snps_map.csv")
# class Testreaction_class(unittest.TestCase):
#
#     def test0_remove_no_effect_SNPs(self):
#         df = pd.read_csv("../data/data_geno_map.csv",header=0,index_col=0)
#         return_value = remove_no_effect_SNPs(df)
#         self.assertEqual(return_value, 0)
#
#     def test1_na_count(self):
#         missing_value_count()


def main():

    choose_snps()
    #unittest.main()
    '''
    height_file = "../data/spline_extract_features/height_predict_withp_spline.csv"
    height_predict,height_deriv,height_deriv2 = pre_process_for_spline_extract_features(height_file)
    height_predict.to_csv("../data/spline_extract_features/height_predict_reformat.csv")
    height_deriv.to_csv(
        "../data/spline_extract_features/height_deriv_reformat.csv")
    height_deriv2.to_csv(
        "../data/spline_extract_features/height_deriv2_reformat.csv")

    la_file = "../data/spline_extract_features/LA_predict_withp_spline.csv"
    la_predict,la_deriv,la_deriv2 = pre_process_for_spline_extract_features(la_file)
    la_predict.to_csv("../data/spline_extract_features/la_predict_reformat.csv")
    la_deriv.to_csv(
        "../data/spline_extract_features/la_deriv_reformat.csv")
    la_deriv2.to_csv(
        "../data/spline_extract_features/la_deriv2_reformat.csv")
    '''

    #read_rdata_to_csv()
    #data_manual_platform_DH = dill.load(open("../data/data_manual_platform_DH", "rb"))
    #plot_raw_data(data_manual_platform_DH)
    # load genotype data and platform phenotype data (DH)
    #data_geno_genotype=dill.load(open("../data/data_geno_genotype","rb"))

    #calculate_SNPs_relationship(snps_df=data_geno_genotype)
    # data_DH_platform = dill.load(open("../data/image_DHline_data", "rb"))
    #
    #data_DH_platform = pd.read_csv("../data/image_DHline_data_after_average_based_on_day_log.csv",header=0,index_col=0)
    # data_DH_platform = pd.read_csv("../data/spline_extract_features/height_predict_withp_spline.csv")
    # generate_a_test_set_by_duplicate_the_data_from_several_genotypes_and_add_noise(
    #     data_DH_platform)

    # field_DH_data =dill.load(open("../data/field_DHline_data",'rb'))
    # gene_field = set(field_DH_data["InbredCode"].unique())
    # gene_platform = set(data_DH_platform["genotype_name"].unique())
    # print(len(gene_field.intersection(gene_platform)))
    # print(len(field_DH_data["InbredCode"].unique()))
    # print(len(data_DH_platform["genotype_name"].unique()))
    # print(len(data_geno_genotype.index))
    #print(data_geno_genotype)
    #print(data_DH_platform)
    #read_rdata_to_csv()
    # average_df = calculate_average_traits_value_by_day(data_DH_platform)
    # transfor_columns = ["LA_Estimated","Height_Estimated","Biomass_Estimated"]
    # log_transformed_df = log_transform(average_df,transfor_columns)
    # log_transformed_df.to_csv("../data/image_DHline_data_after_average_based_on_day.csv")
    #remove_no_effect_SNPs(data_DH_platform,data_geno_genotype)
    #gene_after_remove=dill.load(open("../data/remove_no_effect_snps","rb"))

    #merge_traits_SNPs(data_DH_platform,gene_after_remove)

    '''
    #read field data
    DH_field_data=pd.read_csv("field_DHline_data.csv",header=0,index_col=0)
    hybrids_field_data = pd.read_csv("field_hybrids_data.csv", header=0, index_col=0)
    # check genotype
    ch1=check_genotype(DH_field_data)
    ch2=check_genotype(hybrids_field_data)
    check_gene = list(set(ch1+ch2))
    print(check_gene)
    # read platform data and remove check genotype from it
    DH_platform_data = pd.read_csv("image_DHline_data.csv",header=0,index_col=0)
    hybrids_platform_data = pd.read_csv("image_hybrids_data.csv",header=0,index_col=0)
    remove_check_genotype(DH_platform_data,check_gene)
    remove_check_genotype(hybrids_platform_data, check_gene)

    data_prepared_for_pca = DH_platform_data[["LA_Estimated","Height_Estimated","Biomass_Estimated","genotype_name","DAS"]]
    # drop na based on LA
    data_prepared_for_pca=data_prepared_for_pca.dropna(subset=["LA_Estimated"])
    print(data_prepared_for_pca)
    sns.lineplot(y="LA_Estimated",x="DAS",data=data_prepared_for_pca,hue='genotype_name',linewidth=0.2,errorbar=None,legend=None)
    plt.savefig("LA.png",dpi=1000)
    plt.show()
    '''
    # from data_analysis import run_PCA
    # run_PCA(data_prepared_for_pca,y_label=DH_platform_data["genotype_name"],color_column="genotype_name",file_name="test")

if __name__ == '__main__':
    main()


