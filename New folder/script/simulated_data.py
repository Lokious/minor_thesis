"""
Thhis script is use for simulated time series data use for RNN/LSTM testing
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

def generate_data():
    """
    create 3D input dataset, with 41 time steps, 418 genotypes and 3 features
    :return: list of dataframe
    """

    features = ['trait_1','trait_2','traits_3']
    genoypes = 418
    time_step = 41
    time_series = []

    for feature in features:
        # create time point(X)
        feature_df = pd.DataFrame()
        for genotype in range(genoypes):
            time = np.array([x for x in range(0, time_step)])
            #print(time)

            # create Y based on logistic function
            Y = []
            ## inilize parameters
            k = random.uniform(0.3, 0.5)
            L = random.randint(30, 60)
            T = 30
            #print(k, L, T)
            for t in time:
                # calculate Y and also add gaussian noise
                Y.append((L / (1 + np.exp(-k * (t - T))))+ random.uniform(0, 1))


            #Y = np.array([(y + random.uniform(0, 3)) for y in Y])
            Y = pd.DataFrame(data=Y,index=range(time_step),columns=[genotype])
            feature_df = pd.concat([feature_df,Y],axis=1)
            # plt.scatter(time, Y)
            # plt.show()

        time_series.append(feature_df)
        #print(time_series)
    return time_series


def return_simulated_dataset():


    input_data = generate_data()

    return input_data


def main():

    input_data = return_simulated_dataset()
    print(input_data)


if __name__ == '__main__':
    main()
