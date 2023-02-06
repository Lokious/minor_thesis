import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def read_and_reformat(file:str):
    platform_data = pd.read_csv(file,header=0,index_col=0)
    platform_data = platform_data.fillna(0.0)
    print(len(platform_data["genotype_name"].unique()))
    #print(platform_data)
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

    new_df_LA.to_csv("../data/df_LA.csv")
    new_df_Height.to_csv("../data/df_Height.csv")
    new_df_Biomass.to_csv("../data/df_Biomass.csv")
    return  new_df_LA,new_df_Height,new_df_Biomass
def clusterinf(data_df):
    from tslearn.clustering import TimeSeriesKMeans
    model = TimeSeriesKMeans(n_clusters=418, metric="dtw", max_iter=10)
    model.fit(data_df.T)
    y = model.predict(data_df.T)
    print(y)



def main():
    LA,Height,Biomass = read_and_reformat(file="../data/image_DHline_data_after_average_based_on_day.csv")

    clusterinf(LA)



if __name__ == '__main__':
    main()