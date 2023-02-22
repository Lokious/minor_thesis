import copy

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
import os

def read_and_reformat(file:str):
    platform_data = pd.read_csv(file,header=0,index_col=0)

    #platform_data = platform_data.fillna(0.0)
    # print(len(platform_data["genotype_name"].unique()))

    platform_data = platform_data[platform_data['genotype_name'].str.startswith('DH')]
    #platform_data.to_csv("../data/image_DHline_data_after_average_based_on_day.csv")
    groups_object = platform_data.groupby('plantid')
    group_number = len(groups_object.groups)
    print("group_num:{}".format(group_number))
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
            #print("time steps:{}".format(time_length))
        #time_series_df_list.append(plant_df)
        plantid_list.append(plant_df["plantid"].unique()[0])
    #print(length)

    new_df_LA = pd.DataFrame(index=time_list,columns=plantid_list)
    new_df_Height = pd.DataFrame(index=time_list, columns=plantid_list)
    new_df_Biomass = pd.DataFrame(index=time_list, columns=plantid_list)

    platform_data.set_index(['plantid','DAS'],inplace=True)
    for i in time_list:
        for column in plantid_list:
            try:
                #print(platform_data.loc[(column,i)])
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
    #print(time_list)

    fillna(new_df_LA, time_list, plantid_list)
    fillna(new_df_Height,time_list,plantid_list)
    fillna(new_df_Biomass,time_list,plantid_list)

    #print(new_df_LA)
    # new_df_LA.to_csv("../data/df_LA.csv")
    # new_df_Height.to_csv("../data/df_Height.csv")
    # new_df_Biomass.to_csv("../data/df_Biomass.csv")
    return new_df_LA,new_df_Height,new_df_Biomass

def fillna(new_df, time_list, plantid_list):

    for i in range(len(time_list)):
    #for i in [1]:
        for column in range(len(plantid_list)):
        #for column in[0]:
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
                        #print("step10")
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
                                #print("fill")
                                break
                            else:
                                move_step += 1
                                continue
                        else:
                            print(i - move_step)
                            print(time_list[i],plantid_list[column])
                            raise ValueError("something wrong")
            if pd.isna(new_df.iloc[i, column]):
                print("NA did not be filled?")
                print(i, column)


def clusterinf(data_df,filename:str,centers):
    plot_data= copy.deepcopy(data_df).T
    # plt.plot(plot_data)
    # plt.show()
    #print(data_df)
    #data_df = data_df["Biomass_Estimated_log_transformed","Height_Estimated_log_transformed","LA_Estimated_log_transformed"]
    # 418 genotypes

    '''
    model = KMeans(n_clusters=418,n_init=10)
    print(data_df.isna().values.any())
    y = model.fit_predict(data_df.T)
    y = pd.DataFrame(data=y,columns=["predict"],index=list(range(1,1201)))
    y.to_csv("clustering_result_kmean_{}.csv".format(filename))
    '''

    # init=? If an ndarray is passed, it should be of shape (n_clusters, ts_size, d) and gives the initial centers.

    initial_center =data_df[centers].T
    initial_center = np.expand_dims(initial_center, axis=2)
    print(initial_center.shape)
    #print(initial_center)

    input_data = np.expand_dims(data_df.T, axis=2)
    model = TimeSeriesKMeans(n_clusters=3, metric="softdtw",max_iter=10,init=initial_center)

    model.fit(input_data)
    y_ts = model.predict(input_data)
    y_ts = pd.DataFrame(data=y_ts, columns=["predict"], index=list(range(1,33)))
    y_ts["plantid"] = plot_data.index
    y_ts.to_csv("../data/clustering_result_tslearn_{}_simulated_nonoise.csv".format(filename))
    return y_ts
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure

def evaluate_clustering(true_labels, predicted_labels,filename):

    predict_df = pd.DataFrame()
    predict_df["predict"] = list(predicted_labels)
    predict_df["true_label"] = list(true_labels)
    predict_df.to_csv("../data/clustering_result_tslearn_{}_simulated_nonoise.csv".format(filename))
    ari = adjusted_rand_score(true_labels, predicted_labels)
    hcv = homogeneity_completeness_v_measure(true_labels, predicted_labels)
    homogeneity,completeness,v_measure = hcv
    print("ari score:{:3f}".format(ari))
    print("homogeneity: {:3f}".format(homogeneity))
    print("completeness: {:3f}".format(completeness))
    print("v_measure: {:3f}".format(v_measure))
    return ari, hcv

def print_clustering_result(input_df,centers,true_label, trait:str):


    predict_label = clusterinf(input_df, "la", centers)
    df = pd.read_csv(
        "../data/simulated_data_6_genotype_4_rep_nonoise.csv", header=0,
        index_col=0)
    result_df = df.merge(predict_label, on="plantid")
    print(result_df)
    #colored based on predicted label


    result_df.to_csv("../data/clustering_result_tslearn_{}_simulate_nonoise.csv".format(trait))
    predict_plot = sns.lineplot(data=result_df, x="DAS", y="{}_Estimated_log_transformed".format(trait),
                 units="plantid",hue="predict",estimator=None,palette=['r','g','b'])
    predict_plot.figure.savefig("{}_predict_cluster_nonoise.png".format(trait))
    plt.show()
    # colored based on genotype
    genotype_plot = sns.lineplot(data=result_df, x="DAS", y="{}_Estimated_log_transformed".format(trait),
                 units="plantid",hue="genotype_name",estimator=None,palette=['r','g','b'])
    genotype_plot.figure.savefig("{}_genotype_cluster_nonoise.png".format(trait))
    plt.show()

    # height_result_kmean = pd.read_csv(
    #     "../data/clustering_result_kmean_{}.csv".format(trait), header=0,
    #     index_col=0)

    # print result

    print("time series clustering, dtw:{}".format(trait))
    evaluate_clustering(true_label, predict_label["predict"], trait)
    # print("kmean clustering:{}".format(trait))
    # evaluate_clustering(true_label, height_result_kmean["predict"])

def main():
    #LA, Height, Biomass = read_and_reformat(file="../data/image_DHline_data_after_average_based_on_day.csv")
    LA, Height, Biomass = read_and_reformat(
        file="../data/simulated_data_6_genotype_4_rep_nonoise.csv")

    # # print(LA)
    df = pd.read_csv(
        "../data/simulated_data_6_genotype_4_rep_nonoise.csv", header=0,
        index_col=0)

    df1 = df[["plantid", "genotype_name"]].drop_duplicates(subset=["plantid"])
    centers = df[["plantid", "genotype_name"]].drop_duplicates(subset=["genotype_name"])["plantid"]
    print("center")
    print(centers)
    true_label = df1['genotype_name']

    # clustering and show result
    print_clustering_result(LA,centers=centers,true_label=true_label, trait="LA",)
    print_clustering_result(Height,centers=centers,true_label=true_label,trait="Height")
    print_clustering_result(Biomass,centers=centers,true_label=true_label,trait="Biomass")

if __name__ == '__main__':
    main()