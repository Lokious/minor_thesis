"""
Thhis script is use for simulated time series data use for RNN/LSTM testing
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import unittest
def generate_data():
    """
    create 3D input dataset, with 41 time steps, 418 genotypes and 3 features
    :return: list of dataframe
    """

    features = ['trait_1','trait_2','trait_3']
    genoypes = 418
    time_step = 41
    repetation = 3 # represent the repetition of genotypes
    time_series = []

    for feature in features:
        # create time point(X)
        feature_df = pd.DataFrame()
        without_noise_df = pd.DataFrame()
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
                Y_without_noise = []
                for t in time:
                    # calculate Y and also add gaussian noise
                    noise = random.uniform(0.1,1)
                    Y_without_noise.append((L / (1 + np.exp(-k * (t - T)))))
                    Y.append((L / (1 + np.exp(-k * (t - T))))+noise)
                Y_df = pd.DataFrame(data=Y,index=range(time_step),columns=[(str(genotype)+"_"+str(rep))])

                Y_without_noise_df = pd.DataFrame(data=Y_without_noise,index=range(time_step),columns=[(str(genotype)+"_"+str(rep))])
                without_noise_df = pd.concat([without_noise_df,Y_without_noise_df],axis=1)
                feature_df = pd.concat([feature_df,Y_df],axis=1)

                # plt.plot(time, Y)
                # plt.show()
        else:
            #print(feature_df.shape)
            #print(feature_df)
            feature_df.to_csv("../data/simulated_data/{}.csv".format(feature))
            # also save the noise use for compare with the model prediction,
            # to test if model can generate the dataset exclude the noise
            without_noise_df.to_csv("../data/simulated_data/{}_without_noise.csv".format(feature))
            time_series.append(feature_df)
        #print(time_series)
    return time_series


def return_simulated_dataset():
    input_data = generate_data()

    return input_data


class Testreaction_Class(unittest.TestCase):

    def test0_generate_data(self):
        time_series_df = generate_data()
        # 3 df in a list
        self.assertEqual(len(time_series_df),3)


def main():
    unittest.main()
if __name__ == '__main__':
    main()
