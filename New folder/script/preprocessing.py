"""
This code is to preprocess the phenotype and genotype data

save the dataframe using dill; csv file tooks more time and memory while reading
"""
import random
import dill
import pyreadr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

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
    # very plant should be measured once per day, for those measured multiple times, calculate mean
    after_average = trait_df.groupby(["plantid","Day"])["LA_Estimated", "Height_Estimated", "Biomass_Estimated"].mean().reset_index()
    print(after_average)
    useful_column=after_average.merge(non_numical_column,on=["plantid","Day"])
    print(useful_column)
    useful_column.to_csv("../data/image_DHline_data_after_average_based_on_day.csv")

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

def remove_no_effect_SNPs(geno_df:pd.DataFrame)->int:

    """
    This function is used to merge SNPs(need to select before merge, there are around 500,000 SNPs and phenotype data
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
        print(i)
        if len(geno_df[snp].unique()) == 1:
            print(geno_df[snp].unique())
            print("drop")
            geno_df.drop(columns=[snp],inplace=True)
        else:
            percentage=geno_df[snp].value_counts(normalize=True)
            percentage_dictionary[snp]=percentage
    print(percentage_dictionary)
    print(len(percentage_dictionary.keys()))

    #save the df after remove snps which is the same for all
    with open ("../data/remove_no_effect_snps","wb") as dillfile:
        dill.dump(geno_df,dillfile)

    return 0

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

    groups_object = DH_platform_data.groupby('Pot')
    group_number=len(groups_object.groups)
    color_list=get_colors(group_number)
    for item in groups_object.groups:
        #group based on plant, every plant related to one line

        #print(item)
        plant = groups_object.get_group(item)
        print(plant[['DAS',"Pot"]])
        # drop duplicate day( ned to to change to average ..)
        # plant = plant.drop_duplicates(subset=['DAS'])
        #print(plant)
        for day in plant.index:
            plant_id = plant.loc[day, 'plantid']
            pot_id = plant.loc[day, 'Pot']
            #check pot id equals plant id
            assert plant_id==pot_id
        else:
            #it has 4 different harvest day
            # print(plant['DAS'])
            # print(plant['LA_Estimated'])
            plt.plot(plant["DAS"],plant['visi'],label=plant.loc[day, 'plantid'],linewidth=0.2)
            plt.axvline(x = plant.loc[list(plant["DAS"].index)[-1],"DAS"], linestyle = '-',linewidth=1)

            #plt.legend()
        # if item==20:
        #     break
    plt.ylabel("Biomass_Estimated")
    plt.xlabel("Days")
    #plt.savefig("Biomass_Estimated.png")
    plt.show()

def main():
    #read_rdata_to_csv()
    #data_manual_platform_DH = dill.load(open("../data/data_manual_platform_DH", "rb"))
    #plot_raw_data(data_manual_platform_DH)
    # load genotype data and platform phenotype data (DH)
    data_geno_genotype=dill.load(open("../data/data_geno_genotype","rb"))
    data_DH_platform = dill.load(open("../data/image_DHline_data", "rb"))
    field_DH_data =dill.load(open("../data/field_DHline_data",'rb'))
    gene_field = set(field_DH_data["InbredCode"].unique())
    gene_platform = set(data_DH_platform["genotype_name"].unique())
    print(len(gene_field.intersection(gene_platform)))
    print(len(field_DH_data["InbredCode"].unique()))
    print(len(data_DH_platform["genotype_name"].unique()))
    print(len(data_geno_genotype.index))
    #print(data_geno_genotype)
    #print(data_DH_platform)
    #read_rdata_to_csv()
    #calculate_average_traits_value_by_day(data_DH_platform)

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




# groups_object = DH_data.groupby('Day')
# for item in groups_object.groups:
#     print(item)
#     day_group = groups_object.get_group(item)
#     print(len(day_group["genotype_name"].unique()))