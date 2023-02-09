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
    repetation = 3 # represent the repetition of genotypes
    time_series = []

    for feature in features:
        # create time point(X)
        feature_df = pd.DataFrame()
        L = random.randint(30, 60)
        for genotype in range(genoypes):
            time = np.array([x for x in range(0, time_step)])
            #print(time)
            ## inilize parameters
            k = random.uniform(0.3, 0.5)
            T = 30
            for rep in range(repetation):
                # create Y based on logistic function
                Y = []
                for t in time:
                    # calculate Y and also add gaussian noise
                    Y.append((L / (1 + np.exp(-k * (t - T))))+random.uniform(0, 1))
                Y_df = pd.DataFrame(data=Y,index=range(time_step),columns=[str(genotype)+"_"+str(rep)])
                feature_df = pd.concat([feature_df,Y_df],axis=1)
                # plt.plot(time, Y)
                # plt.show()
        print(feature_df.shape)
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
