import pandas as pd
import numpy as np
import glob

def read_reformat(file_directory:str,file_name_end_with=".txt"):
    files = glob.glob("{}*{}".format(file_directory,file_name_end_with))
    print(files)
    biomass_df =pd.DataFrame()
    # read and merge to one csv
    for file in files:
        df_new = pd.read_table(file,header=0,index_col=0,sep=" ")
        biomass_df = pd.concat([biomass_df,df_new])
        print(biomass_df)
    else:
        biomass_df.to_csv("biomass.csv")

    #separate based on environment
    biomass_df.groupby("Env")
    
def main():
    read_reformat("./data/")
if __name__ == '__main__':
    main()
    