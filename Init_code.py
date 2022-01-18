"""
LOADING LIBRARY ----
"""
import logging
import pickle 
import os
import pandas as pd
import numpy as np
import qgrid
import datetime 
import talib
import pandas as pd
from functools import reduce
import pdb
from dateutil.parser import parse
import warnings
warnings.filterwarnings('ignore')
curr_dir = os.getcwd()
print('Your Curr dir is:\n\t\t',curr_dir)

"""
CREATING FOLDER AT SET DIRECTORY ----
"""
#CREATING DEFAULT FOLDERS FOR TEMPRARRY DATA STORING ---
folder_list=['input_data','temp_data','output_data','data_repos']
for iin in folder_list:
    if not os.path.exists(f'{iin}'):
        os.makedirs(f'{iin}') 
        
"""
WRITING PANDAS DATAFRAME ONE BELOW OTHER USING FOR-LOOP IN ONE SHEET ----
"""
dfs = [df1, df2, df3]
startrow = 0
with pd.ExcelWriter('output.xlsx') as writer:
    for df in dfs:
        df.to_excel(writer, engine="xlsxwriter", startrow=startrow)
        startrow += (df.shape[0] + 2)
        

"""       
WRITTING MULTIPLE DATAFRAMEIN IN MULTIPLE SHEET IN SAME EXCEL SHEET ----
"""
import pandas as pd
df_list = df_list
writer = pd.ExcelWriter('temp_data/rep.xlsx', engine='xlsxwriter')
for i in df_list:
    run_cmd = f"{i}.to_excel(writer, sheet_name='{i}')";exec(run_cmd)
writer.save()




  
  
 
