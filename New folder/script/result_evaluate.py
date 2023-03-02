import numpy as np
import pandas as pd
import copy


def merge_snp_with_region(snp_file:str,snp_map_file:str="../data/data_geno_map.csv"):
    snp_df = pd.read_csv(snp_file,header=0,index_col=0)
    snp_df = snp_df.rename(columns={'0':"rownames"})
    map_df = pd.read_csv(snp_map_file,header=0,index_col=0)
    print(map_df)

    snp_df = snp_df.merge(map_df,on="rownames")
    print(snp_df)
    snp_df = snp_df.rename(columns={"rownames":"snps"})
    snp_df.to_csv(snp_file)

def merge_snp_to_test_result(snp_file:str,test_result:str):
    snp_df = pd.read_csv(snp_file,header=0,index_col=0)
    test_result_df = pd.read_csv(test_result,header=0,index_col=0)
    test_result_df["snps"]=test_result_df.index
    test_result_df.reset_index(inplace=True)
    print(test_result_df)
    snp_df_result = snp_df.merge(test_result_df,on="snps")
    snp_df_result.dropna(inplace=True)
    print(snp_df_result)
def main():
    #step1
    #merge_snp_with_region(snp_file="../result/log_spline_predict_result_lstm_DH_LA_and_Height/possiable_related_snps_log.csv")
    merge_snp_to_test_result(snp_file="../result/log_spline_predict_result_lstm_DH_LA_and_Height/possiable_related_snps_log.csv",
                             test_result="../result/log_spline_predict_result_lstm_DH_LA_and_Height/LSTM_test_result_log.csv")
if __name__ == '__main__':
    main()
