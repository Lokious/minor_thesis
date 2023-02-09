import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans

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
    for i in range(len(time_list)):
        print(i)
        for column in range(len(plantid_list)):
            if (pd.isna(new_df_LA.iloc[i,column])):
                try:
                    if i == 0:
                        # if the NA at the first time step
                        new_df_LA.iloc[i, column] = 0
                    elif i == len(time_list)-1:
                        # if the NA at the last time step
                        new_df_LA.iloc[i, column] = new_df_LA.iloc[i - 1, column]
                    else:

                        print("column {}".format(column))
                        move_step = 1
                        if (pd.isna(new_df_LA.iloc[i-move_step,column]) or pd.isna(new_df_LA.iloc[i+move_step,column]))==False:
                            print(new_df_LA.iloc[i-move_step,column],new_df_LA.iloc[i+move_step,column])
                            average_value = (new_df_LA.iloc[i-move_step,column] + new_df_LA.iloc[i+move_step,column])/2
                            print("average {}".format(average_value))
                            new_df_LA.iloc[i,column] = average_value
                        else:
                            raise ValueError("nan in the besides cell(s)")
                except:
                    # if there are two or more continues NA
                    print("not the first nor the last")
                    while (i+move_step < len(time_list)-1):
                        try:
                            move_step += 1
                            print(move_step+i)
                            new_df_LA.iloc[i, column] = new_df_LA.iloc[
                            i - 1, column] + ((new_df_LA.iloc[i + move_step, column] - new_df_LA.iloc[
                            i - 1, column]) / (move_step+1))
                            new_df_LA.iloc[i+1, column] = new_df_LA.iloc[
                            i - 1, column] + 2*((new_df_LA.iloc[i + move_step, column] - new_df_LA.iloc[
                            i - 1, column]) / (move_step+1))
                            break
                        except:
                            continue
                    else:
                        print("all following values are na, fill in the following cell with the last non-na value")
                        move_step = 1
                        while (i-move_step) > 0:
                            if pd.isna(new_df_LA.iloc[i - move_step, column])==False:
                                new_df_LA.iloc[i, column] = new_df_LA.iloc[i - move_step, column]
                                break
                            else:
                                move_step +=1
                                continue
                    print("fill with average")
    for i in range(len(time_list)):
        print(i)
        for column in range(len(plantid_list)):
            if (pd.isna(new_df_Height.iloc[i,column])):
                try:
                    if i == 0:
                        # if the NA at the first time step
                        new_df_Height.iloc[i, column] = 0
                    elif i == len(time_list)-1:
                        # if the NA at the last time step
                        new_df_Height.iloc[i, column] = new_df_Height.iloc[i - 1, column]
                    else:

                        print("column {}".format(column))
                        move_step = 1
                        if (pd.isna(new_df_Height.iloc[i-move_step,column]) or pd.isna(new_df_Height.iloc[i+move_step,column]))==False:
                            print(new_df_Height.iloc[i-move_step,column],new_df_Height.iloc[i+move_step,column])
                            average_value = (new_df_Height.iloc[i-move_step,column] + new_df_Height.iloc[i+move_step,column])/2
                            print("average {}".format(average_value))
                            new_df_Height.iloc[i,column] = average_value
                        else:
                            raise ValueError("nan in the besides cell(s)")
                except:
                    # if there are two or more continues NA
                    print("not the first nor the last")
                    while (i+move_step < len(time_list)-1):
                        try:
                            move_step += 1
                            print(move_step+i)
                            new_df_Height.iloc[i, column] = new_df_Height.iloc[
                            i - 1, column] + ((new_df_Height.iloc[i + move_step, column] - new_df_Height.iloc[
                            i - 1, column]) / (move_step+1))
                            new_df_Height.iloc[i+1, column] = new_df_Height.iloc[
                            i - 1, column] + 2*((new_df_Height.iloc[i + move_step, column] - new_df_Height.iloc[
                            i - 1, column]) / (move_step+1))
                            break
                        except:
                            continue
                    else:
                        print("all following values are na, fill in the following cell with the last non-na value")
                        move_step = 1
                        while (i-move_step) > 0:
                            if pd.isna(new_df_Height.iloc[i - move_step, column])==False:
                                new_df_Height.iloc[i, column] = new_df_Height.iloc[i - move_step, column]
                                break
                            else:
                                move_step +=1
                                continue
                    print("fill with average")
    for i in range(len(time_list)):
        print(i)
        for column in range(len(plantid_list)):
            if (pd.isna(new_df_Biomass.iloc[i,column])):
                try:
                    if i == 0:
                        # if the NA at the first time step
                        new_df_Biomass.iloc[i, column] = 0
                    elif i == len(time_list)-1:
                        # if the NA at the last time step
                        new_df_Biomass.iloc[i, column] = new_df_Biomass.iloc[i - 1, column]
                    else:

                        print("column {}".format(column))
                        move_step = 1
                        if (pd.isna(new_df_Biomass.iloc[i-move_step,column]) or pd.isna(new_df_Biomass.iloc[i+move_step,column]))==False:
                            print(new_df_Biomass.iloc[i-move_step,column],new_df_Biomass.iloc[i+move_step,column])
                            average_value = (new_df_Biomass.iloc[i-move_step,column] + new_df_Biomass.iloc[i+move_step,column])/2
                            print("average {}".format(average_value))
                            new_df_Biomass.iloc[i,column] = average_value
                        else:
                            raise ValueError("nan in the besides cell(s)")
                except:
                    # if there are two or more continues NA
                    print("not the first nor the last")
                    while (i+move_step < len(time_list)-1):
                        try:
                            move_step += 1
                            print(move_step+i)
                            new_df_Biomass.iloc[i, column] = new_df_Biomass.iloc[
                            i - 1, column] + ((new_df_Biomass.iloc[i + move_step, column] - new_df_Biomass.iloc[
                            i - 1, column]) / (move_step+1))
                            new_df_Biomass.iloc[i+1, column] = new_df_Biomass.iloc[
                            i - 1, column] + 2*((new_df_Biomass.iloc[i + move_step, column] - new_df_Biomass.iloc[
                            i - 1, column]) / (move_step+1))
                            break
                        except:
                            continue
                    else:
                        print("all following values are na, fill in the following cell with the last non-na value")
                        move_step = 1
                        while (i-move_step) > 0:
                            if pd.isna(new_df_Biomass.iloc[i - move_step, column])==False:
                                new_df_Biomass.iloc[i, column] = new_df_Biomass.iloc[i - move_step, column]
                                break
                            else:
                                move_step +=1
                                continue
                    print("fill with average")

    print(new_df_LA)
    # new_df_LA.to_csv("../data/df_LA.csv")
    # new_df_Height.to_csv("../data/df_Height.csv")
    # new_df_Biomass.to_csv("../data/df_Biomass.csv")
    return new_df_LA,new_df_Height,new_df_Biomass

def clusterinf(data_df):

    #print(data_df)
    #data_df = data_df["Biomass_Estimated_log_transformed","Height_Estimated_log_transformed","LA_Estimated_log_transformed"]
    # 418 genotypes
    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=3,n_init=20)

    y = model.fit_predict(data_df.T)
    y = pd.DataFrame(data=y,columns=["predict"],index=list(range(1200)))
    # print(y)
    # print(y["predict"].unique())
    y.to_csv("clustering_result.csv")

def main():
    LA,Height,Biomass = read_and_reformat(file="../data/image_DHline_data_after_average_based_on_day.csv")
    # print("LA")
    # print(LA)
    clusterinf(LA)



if __name__ == '__main__':
    main()