"""
Thhis script is use for simulated time series data use for RNN/LSTM testing
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

def generate_data():
    # create time point(X)
    time = np.array([x for x in range(0, 60)])
    #print(time)

    # create Y based on logistic function
    Y = []
    ## inilize parameters
    k = random.uniform(0.3, 0.5)
    L = random.randint(30, 60)
    T = 30
    #print(k, L, T)
    for t in time:
        Y.append(L / (1 + np.exp(-k * (t - T))))

    # add gaussian noise
    Y = np.array([(y + random.uniform(0, 3)) for y in Y])

    # plt.scatter(time, Y)
    # plt.show()
    time_series = pd.DataFrame(data=Y)
    #print(time_series)
    return time_series


def return_simulated_dataset():
    input_data = pd.DataFrame()
    for i in range(418):
        new_column = generate_data()
        input_data = pd.concat([input_data,new_column],axis=1)
    input_data.columns=list(range(0,418))
    return input_data


def main():

    input_data = return_simulated_dataset()
    print(input_data)


if __name__ == '__main__':
    main()
