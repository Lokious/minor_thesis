import random
import copy
import unittest
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA

from preprocessing import get_colors


def run_PCA(datasets, y_label, color_column, file_name=""):
    """

    :param self:
    :param datasets:
    :param y_label:
    :param color_column:
    :param file_name:
    :return:
    """
    # show pca result based on methylation type
    data_with_site = copy.deepcopy(datasets).drop(
        columns=[color_column])
    def label_with_color_type(color_column):
        """return color map based on the column used for color the data points"""
        map_dictionary = {}

        colours = get_colors(len(datasets[color_column].unique()))
        for type in datasets[color_column].unique():
            map_dictionary[type] = colours.pop(0)
        colour_label = datasets[color_column].map(map_dictionary)
        print(colour_label)
        return colour_label ,map_dictionary

    colour_label, map_dictionary = label_with_color_type(color_column)
    V = []
    PC = []
    # the most is 522 for pca
    for i in range(len(data_with_site.columns)):
        PC.append("PC" + str(i + 1))
        V.append("V" + str(i + 1))
        if i == min(521 ,(len(datasets.index ) -1)):
            break

    pca_fit = PCA(n_components=len(PC) ,random_state=42).fit(data_with_site)
    pca_loadings = pd.DataFrame(pca_fit.components_.T,
                                index=data_with_site.columns, columns=V)
    # show the lodings
    V1_PCA = pd.DataFrame(data=pca_loadings.abs(),
                          index=pca_loadings.index,
                          columns=["V1" ])

    V1_PCA['V_sum'] = V1_PCA.apply(lambda x: (x.abs()).sum(), axis=1)
    V1_PCA = V1_PCA.sort_values(by=["V_sum"], ascending=False)
    sns.barplot(data=V1_PCA.iloc[:30], x="V_sum", y=V1_PCA.index[:30]).set(
        title='v1  loading value of PCA')
    plt.tight_layout()
    plt.savefig("../of_v1loading_value_of_PCA_{}".format(file_name))

    plt.close()
    pca_df = pd.DataFrame(pca_fit.fit_transform(data_with_site)
                          ,index=data_with_site.index, columns=PC)

    fig, ax = plt.subplots()
    plt.scatter(pca_df.PC1, pca_df.PC2, c=colour_label, s=3)

    # show explained variance
    handle_list = []
    for key in map_dictionary.keys():
        handle_list.append \
            (mpatches.Patch(color=map_dictionary[key], label=key))
    # ax.legend(
    #     handles=handle_list ,loc="upper right" ,title="Type")

    # plt.scatter(pca_df.PC1, pca_df.PC2,  s=5,c=colour_label)
    print(pca_fit.explained_variance_ratio_)
    plt.xlabel("pc1({:.2f}%)".format
        (round(pca_fit.explained_variance_ratio_[0 ] *100) ,2))
    plt.ylabel("pc2({:.2f}%)".format
        (round(pca_fit.explained_variance_ratio_[1 ] *100) ,2))
    plt.title("PCA coloured by gene type")
    plt.savefig(
        "../pca_for PC1 PC2{}.png".format
            (file_name),dpi=1000)
    plt.clf()
    plt.close()
    plt.plot(list(range(1, len(pca_df.columns) + 1)),
             pca_fit.explained_variance_ratio_, '-ro')
    plt.ylabel('Proportion of Variance Explained_{}'.format(file_name))
    plt.xlabel("components")
    plt.savefig(
        '../Proportion of Variance Explained_{}.svg'.format(file_name))

    plt.clf()
    plt.plot(list(range(1, len(pca_df.columns) + 1)),
             np.cumsum(pca_fit.explained_variance_ratio_), '-o')
    plt.ylabel \
        ('Cumulative Proportion of Variance Explained_{}'.format(file_name))
    plt.xlabel("components")
    plt.savefig(
        '../Cumulative Proportion of Variance Explained_{}.svg'.format
            (file_name))

    plt.clf()

    return pca_df
def main():

    print("0")
if __name__ == "__main__":
    main()
