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

"""
UNZIP THE ZIP FILE ---
"""
zip_folder = os.listdir('Derrivative_histData/')
zip_folder

import zipfile
def un_zipFiles(path):
    files=os.listdir(path)
    for file in files:
        if file.endswith('.zip'):
            filePath=path+'/'+file
            zip_file = zipfile.ZipFile(filePath)
            for names in zip_file.namelist():
                zip_file.extract(names,path)
            zip_file.close() 
un_zipFiles('DTA/input_data/')


"""
TARGET ANALYSIS WITH SINGLE/MULTIPLE (CONCATENATE) VARIABLES
"""

def get_ConctargetAnalysis(data_seen,conc_cols,target):
    
    def concat_columns(temp_df,xlab):
        columns = xlab #=['ncb','age_of_vehicle'] # grouping of 
        columns_conc = ("|".join(columns))
        temp_df[f'{columns_conc}_combi'] = temp_df[columns].astype(str).astype(str).apply('|'.join, axis=1)
        return(temp_df)
    
    def rename_multiindex(qr):
        level_one = qr.columns.get_level_values(0).astype(str)
        level_two = qr.columns.get_level_values(1).astype(str)
        qr.columns = level_one +'_'+ level_two
        return(qr)
    
    #conc_cols = ['Experience','Source of Hire','Previous Industry','Age Bucket','Gender','Marital Status']
    temp_df = concat_columns(data_seen,conc_cols)
    col_list = temp_df.columns
    icol  = col_list[-1]
    ti =  temp_df.groupby([f'{icol}',f'{target}']).size().reset_index().rename(columns={0:'event'}).pivot(index=f'{icol}',columns=f'{target}')
    ti = ti.reset_index()
    ti = rename_multiindex(ti)
    ti = ti.replace(np.nan,0)
    ti['TOTPol'] =  ti['event_0'] + ti['event_1']
    ti['event%'] = round((ti['event_1']/ti['TOTPol'])*100,2)
    ti['nevent%'] = round((ti['event_0']/ti['TOTPol'])*100,2)
    ti['cum_tot'] =  round((ti['TOTPol']/ti['TOTPol'].sum())*100,2)
    return(ti)



"""
XLWINGS EXCEL CODE ---
"""
import xlwings as xw # loading library

wb = xw.Book()# creating new workbook
sht =  wb.sheets[0]# initiating sheet-01 to write
sht.clear_contents()# clear the content of sheet
sht.range("A1").value = df# wrtting data frame on to sht sheet
  
"""
Pivot Table Visualization 
"""

import pandas as pd
from pivottablejs import pivot_ui
pivot_ui(temp)
 
