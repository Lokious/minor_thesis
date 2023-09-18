import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy


def merge_snp_with_region(snp_file:str,snp_map_file:str="../data/data_geno_map.csv"):
    """
    merge file: snps types with it's region on chr, print and save the df after merging
    :param snp_file: str, file name(with directory）
    :param snp_map_file: str, ../data/data_geno_map.csv
    :return: None
    """

    snp_df = pd.read_csv(snp_file,header=0,index_col=0)
    snp_df = snp_df.rename(columns={'0':"rownames"})
    map_df = pd.read_csv(snp_map_file,header=0,index_col=0)
    print("####printing ../data/data_geno_map.csv####")
    print(map_df)
    snp_df = snp_df.merge(map_df,on="rownames")
    print(snp_df)
    snp_df = snp_df.rename(columns={"rownames":"snps"})
    snp_df.to_csv(snp_file)


def merge_snp_to_test_result(snp_file:str,test_result:str):
    """
    Merge Ground Truth with prediction result, calculate the accuracy

    :param snp_file: str, file name(with directory）,include region, save from merge_snp_with_region
    :param test_result: str, file name(with directory）, test result from model
    :return:
    """
    snp_df = pd.read_csv(snp_file,header=0,index_col=0)
    test_result_df = pd.read_csv(test_result,header=0,index_col=0)
    test_result_df["snps"]=test_result_df.index
    test_result_df.reset_index(inplace=True,drop=True)
    print(test_result_df)
    snp_df_result = snp_df.merge(test_result_df,on="snps")
    snp_df_result.dropna(inplace=True)

    snp_df_result["accuracy"]=(snp_df_result["TP"]+snp_df_result["TN"])/(snp_df_result["TP"]+snp_df_result["TN"]+snp_df_result
                                                                         ["FP"]+snp_df_result["FN"])
    print(snp_df_result)
    save_dir = "/".join(snp_file.split("/")[:-1])
    print(save_dir)
    snp_df_result.to_csv("{}/result.csv".format(save_dir))


def data_imbalance(df_file = "../result/log_spline_predict_result_lstm_DH_LA_and_Height/LSTM_test_result_log.csv",snp_file="../data/data_geno_map.csv"):
    """
    For imbalanced data, calculate accuracy and MCC, save the result to the original result file
    :return: None
    """
    snp_df = pd.read_csv(snp_file,header=0,index_col=0)
    snp_df["snps"] = snp_df.index
    print(snp_df)
    snp_df.reset_index(inplace=True, drop=True)
    df = pd.read_csv(df_file,header=0,index_col=0)
    df["snps"]=df.index
    df.reset_index(inplace=True,drop=True)

    print(df)
    df["label0"] = df["TN"] + df["FP"]
    df["label1"] = df["TP"] + df["FN"]
    df["accuracy"]=(df["TP"]+df["TN"])/(df["TP"]+df["TN"]+df["FP"]+df["FN"])

    # MCC (TP*TN – FP*FN) / √(TP+FP)(TP+FN)(TN+FP)(TN+FN)
    df["mcc"] = (df["TP"]*df["TN"] - df["FP"]*df["FN"])/np.sqrt((df["TP"]+df["FP"])*(df["TP"]+df["FN"])*(df["TN"]+df["FP"])*(df["TN"]+df["FN"]))
    df["mark"] = 0
    result_df = pd.read_csv("../result/log_spline_predict_result_lstm_DH_LA_and_Height/result.csv")
    snp_list = list(result_df["snps"])
    df.loc[df["snps"].isin(snp_list), 'mark'] = 1

    df = snp_df.merge(df,on="snps")

    df.to_csv(df_file)


def imbalance_data_plot():
    """
    For imbalanced data, plot the percentage of label 1 and 0
    :return:
    """
    import seaborn as sns
    #data_imbalance()
    result_df = pd.read_csv("../result/log_spline_predict_result_lstm_DH_LA_and_Height/LSTM_test_result_log.csv",header=0,index_col=0)
    group_object = result_df.groupby("chr")
    for group_name in group_object.groups:
        group_df = group_object.get_group(group_name)
        group_df.loc[:,"label1_percentage"] = (group_df["label1"]/(group_df["label1"] +group_df["label0"]))
        group_df.loc[:,"label0_percentage"] = (group_df["label0"] / (
                    group_df["label1"] + group_df["label0"]))
        print(group_df["label1_percentage"])
        '''
        fig, ax1 = plt.subplots()
        ax1.bar(x=group_df["pos"],height=group_df["label1_percentage"],color="blue",label="label1",alpha=.7)
        plt.show()
        ax2=ax1.twinx()
        plt.show()
        ax2.bar(group_df["pos"],group_df["accuracy"],color="red",label="accuracy")
        ax3 = ax2.twinx()
        ax3.bar(group_df["pos"], group_df["label0_percentage"], color="orange",
                label="label0")
        '''
        raise EnvironmentError("Something wrong with the plot, need to check!!")
        fig, ax1 = plt.subplots()

        sns.barplot(data=group_df,x=group_df["pos"],y="label1_percentage",color="blue",label="label1",ax=ax1).axhline(0.6,)
        sns.barplot(data=group_df, x=group_df["pos"], y="label0_percentage", color="orange",label="label0",ax=ax1).axhline(0.4,)
        # plt.savefig("../result/log_spline_predict_result_lstm_DH_LA_and_Height/chr{}.png".format(group_name),dpi=800)
        sns.scatterplot(data=group_df, x=group_df["pos"], y="accuracy",
                        color="red", ax=ax1)
        handles, labels = ax1.get_legend_handles_labels()
        plt.legend(handles, labels)
        #plt.savefig("chr{}.png".format(group_name),dpi=600)
        plt.show()
        plt.clf()

def correlation_between_snps_and_features():

    import dill
    snp_list = list(pd.read_csv("../result/log_spline_predict_result_lstm_DH_LA_and_Height_with_no_rep_genotype/result.csv",header=0,index_col=0)["snps"])
    for snp in snp_list:
        input_X = dill.load(open(
            "../data/input_data/spline_predict_input_average/input_x".format(
                snp), "rb"))
        input_y = dill.load(open(
            "../data/input_data/spline_predict_input_average/input_Y_{}".format(
                snp), "rb"))
        print(input_X)
        print(input_y)

def main():
    imbalance_data_plot()
    # #step1
    # merge_snp_with_region(snp_file="../result/log_spline_predict_result_lstm_DH_LA_and_Height_with_no_rep_genotype/possiable_related_snps_log.csv")
    # merge_snp_to_test_result(snp_file="../result/log_spline_predict_result_lstm_DH_LA_and_Height_with_no_rep_genotype/possiable_related_snps_log.csv",
    #                          test_result="../result/log_spline_predict_result_lstm_DH_LA_and_Height_with_no_rep_genotype/LSTM_test_result_log.csv")
    '''
    df1_average = pd.read_csv("../result/log_spline_predict_result_lstm_DH_LA_and_Height_with_no_rep_genotype/result.csv",header=0,index_col=0)
    df2_no_average = pd.read_csv("../result/log_spline_predict_result_lstm_DH_LA_and_Height/result.csv",header=0,index_col=0)
    overlap_df = df2_no_average.merge(df1_average,on="snps")
    print(overlap_df)
    overlap_df.to_csv("result_check.csv")

    '''
    # qtl_field_result_region = pd.read_excel("../../GWAS_DHs_landraces/41467_2020_18683_MOESM5_ESM.xlsx")[["Start reg.(bp)","End reg.(bp)"]].astype('int64')
    #
    # qtl_field_result_region = qtl_field_result_region.sort_values(by=["Start reg.(bp)"])
    # print(qtl_field_result_region[["Start reg.(bp)", "End reg.(bp)"]])
    # result_df = pd.read_csv("../result/log_spline_predict_result_lstm_DH_LA_and_Height/result.csv")
    # count = 0
    # for i in result_df.index:
    #     pos_i = result_df.loc[i,"pos"]
    #     for row in qtl_field_result_region.index:
    #         start = qtl_field_result_region.loc[row,"Start reg.(bp)"]
    #         end = qtl_field_result_region.loc[row,"End reg.(bp)"]
    #         if (pos_i >= start) and (pos_i <= end):
    #             print(pos_i)
    #             count += 1
    #             break
    #         elif pos_i > start:
    #             break
    # print(count)
if __name__ == '__main__':
    main()
