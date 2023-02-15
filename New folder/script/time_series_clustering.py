import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
import os
def read_and_reformat(file:str):
    platform_data = pd.read_csv(file,header=0,index_col=0)
    platform_data = platform_data.fillna(0.0)
    # print(len(platform_data["genotype_name"].unique()))
    # print(platform_data)
    platform_data.to_csv("../data/image_DHline_data_after_average_based_on_day.csv")
    groups_object = platform_data.groupby('plantid')
    group_number = len(groups_object.groups)

    length = 0
    time_list = []
    plantid_list = []
    for item in groups_object.groups:
        # group based on plant, every plant related to one line
        plant_df = groups_object.get_group(item)
        plant_df.set_index("DAS",inplace=True)
        #print(plant_df)
        time_length = len(plant_df.index)
        if time_length > length:
            length = time_length
            time_list = list(plant_df.index)
        #time_series_df_list.append(plant_df)
        plantid_list.append(plant_df["plantid"].unique()[0])

    new_df_LA = pd.DataFrame(index=time_list,columns=plantid_list)
    new_df_Height = pd.DataFrame(index=time_list, columns=plantid_list)
    new_df_Biomass = pd.DataFrame(index=time_list, columns=plantid_list)

    platform_data.set_index(['plantid','DAS'],inplace=True)
    for i in time_list:
        for column in plantid_list:
            try:
                new_df_LA.loc[i,column] = platform_data.loc[(column,i),"LA_Estimated_log_transformed"]
            except:
                new_df_LA.loc[i, column] = np.nan
            try:
                new_df_Height.loc[i,column] = platform_data.loc[(column,i),"Height_Estimated_log_transformed"]
            except:
                new_df_Height.loc[i, column] = np.nan
            try:
                new_df_Biomass.loc[i,column] = platform_data.loc[(column,i),"Biomass_Estimated_log_transformed"]
            except:
                new_df_Biomass.loc[i, column] = np.nan

    # print(new_df_LA)
    # print(new_df_Height)
    # print(new_df_Biomass)
    #use the average to fill in the NA
    print(time_list)
    fillna(new_df_LA, time_list, plantid_list)
    fillna(new_df_Height,time_list,plantid_list)
    fillna(new_df_Biomass,time_list,plantid_list)

    #print(new_df_LA)
    new_df_LA.to_csv("../data/df_LA.csv")
    new_df_Height.to_csv("../data/df_Height.csv")
    new_df_Biomass.to_csv("../data/df_Biomass.csv")
    return new_df_LA,new_df_Height,new_df_Biomass

def fillna(new_df, time_list, plantid_list):
    for i in range(len(time_list)):
        for column in range(len(plantid_list)):
            if pd.isna(new_df.iloc[i, column]):
                try:
                    if i == 0:
                        # if the NA at the first time step
                        new_df.iloc[i, column] = 0
                    elif i == len(time_list) - 1:

                        # if the NA at the last time step
                        new_df.iloc[i, column] = new_df.iloc[
                            i - 1, column]
                    else:

                        # print("column {}".format(column))
                        move_step = 1
                        if not (pd.isna(new_df.iloc[
                                            i - move_step, column]) or pd.isna(
                            new_df.iloc[
                                i + move_step, column])):
                            average_value = (new_df.iloc[
                                                 i - move_step, column] +
                                             new_df.iloc[
                                                 i + move_step, column]) / 2
                            # print("average {}".format(average_value))
                            new_df.iloc[i, column] = average_value
                        else:
                            raise ValueError("nan in the besides cell(s)")
                except:
                    # if there are two or more continues NA
                    move_step = 1
                    while (i + move_step < len(time_list) - 1):
                        # print("step10")
                        move_step += 1
                        if not (pd.isna(new_df.iloc[i - 1, column]) or pd.isna(
                                new_df.iloc[
                                    i + move_step, column])):
                            new_df.iloc[i, column] = new_df.iloc[
                                        i - 1, column] + ((
                                                              new_df.iloc[
                                                                      i + move_step, column] -
                                                              new_df.iloc[
                                                                      i - 1, column]) / (
                                                                  move_step + 1))
                            new_df.iloc[i + 1, column] = new_df.iloc[
                                                                i - 1, column] + 2 * (
                                                                        (
                                                                                new_df.iloc[
                                                                                        i + move_step, column] -
                                                                                new_df.iloc[
                                                                                        i - 1, column]) / (
                                                                                    move_step + 1))
                            break
                    else:
                        # print("all following values are na, fill in the following cell with the last non-na value")
                        move_step = 1
                        while (i - move_step) > 0:
                            if not pd.isna(new_df.iloc[
                                               i - move_step, column]):
                                new_df.iloc[i, column] = new_df.iloc[
                                    i - move_step, column]
                                print("fill")
                                break
                            else:
                                move_step += 1
                                continue
                        else:
                            raise ValueError("something wrong")
            if pd.isna(new_df.iloc[i, column]):
                print("NA did not be filled?")
                print(i, column)


def clusterinf(data_df,filename:str):

    #print(data_df)
    #data_df = data_df["Biomass_Estimated_log_transformed","Height_Estimated_log_transformed","LA_Estimated_log_transformed"]
    # 418 genotypes


    model = KMeans(n_clusters=418,n_init=10)
    print(data_df.isna().values.any())
    y = model.fit_predict(data_df.T)
    y = pd.DataFrame(data=y,columns=["predict"],index=list(range(1,1201)))
    y.to_csv("clustering_result_kmean_{}.csv".format(filename))

    model = TimeSeriesKMeans(n_clusters=418, metric="dtw", max_iter=10)
    model.fit(data_df.T)
    y_ts = model.predict(data_df.T)
    y_ts = pd.DataFrame(data=y_ts, columns=["predict"], index=list(range(1,1201)))
    y_ts.to_csv("clustering_result_tslearn_{}.csv".format(filename))

from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure

def evaluate_clustering(true_labels, predicted_labels):
    ari = adjusted_rand_score(true_labels, predicted_labels)
    hcv = homogeneity_completeness_v_measure(true_labels, predicted_labels)
    homogeneity,completeness,v_measure = hcv
    print("ari score:{:3f}".format(ari))
    print("homogeneity: {:3f}".format(homogeneity))
    print("completeness: {:3f}".format(completeness))
    print("v_measure: {:3f}".format(v_measure))
    return ari, hcv

def print_clustering_result(true_label,trait="height"):

    height_result_tslearn = pd.read_csv("../data/clustering_result_tslearn_{}.csv".format(trait),header=0,index_col=0)
    height_result_kmean = pd.read_csv(
        "../data/clustering_result_kmean_{}.csv".format(trait), header=0,
        index_col=0)

    # print result
    print()
    print("time series clustering, dtw:{}".format(trait))
    evaluate_clustering(true_label,height_result_tslearn["predict"])
    print("kmean clustering:{}".format(trait))
    evaluate_clustering(true_label, height_result_kmean["predict"])

def main():
    # LA, Height, Biomass = read_and_reformat(file="../data/image_DHline_data_after_average_based_on_day.csv")
    # # print("LA")
    # # print(LA)
    # clusterinf(Height,"height")
    # clusterinf(Biomass, "biomass")
    # clusterinf(LA, "LA")
    df = pd.read_csv(
        "../data/image_DHline_data_after_average_based_on_day.csv", header=0,
        index_col=0)
    df1 = df[["plantid", "genotype_name"]].drop_duplicates(subset=["plantid"])
    true_label = df1['genotype_name'].astype('category').cat.codes
    print_clustering_result(true_label, trait="la")
    print_clustering_result(true_label,trait="height")
    print_clustering_result(true_label, trait="biomass")
if __name__ == '__main__':
    main()