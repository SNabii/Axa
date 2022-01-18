# Base Library ---
import os
import pandas as pd 
import numpy as np
import qgrid
import scipy.stats as stats
from datetime import datetime
import time
from dateutil import parser

# Library Stats ---
from scipy.stats import norm, skew, kurtosis, boxcox #for some statistics





# 1. BASE FUNCTIONS :

def get_ctime(txt_pass):
    print(f'*{txt_pass} IS->>****,{datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")}****<<')

t_stamp = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")




def gview(data):
    print('DIMENSION',data.shape)
    return(qgrid.show_grid(data,show_toolbar=True,grid_options={'forceFitColumns': False,'highlightSelectedCell': True,
        'highlightSelectedRow': True}))

def get_stime():
    import time
    stime = time.time()
    return(stime)

def seq(start, stop, step=1):
    n = int(round((stop - start)/float(step)))
    if n > 1:
        return([start + step*i for i in range(n+1)])
    elif n == 1:
        return([start])
    else:
        return([])
    
def get_partialcol_match(final_df,txt):
    date_colist = final_df[final_df.columns[final_df.columns.to_series().str.contains(f'{txt}')]].columns
    date_colist = date_colist.tolist()
    return(date_colist)     
    
def date_formatter(pass_date):
    pass_date = str(pass_date)
    if 'nan' not in pass_date:
        #return(1)
        if '-' in pass_date:
            return(parser.parse(pass_date).strftime("%d-%m-%Y"))
        if '/' in pass_date:
            return(parser.parse(pass_date).strftime("%d/%m/%Y"))
    else:
        return(np.nan) 
    
    
    
# Basic data cleaning
def data_basic_clean(fsh):
        fsh.columns = [c.strip() for c in fsh.columns]
        fsh.columns = [c.replace(' ', '_') for c in fsh.columns]
        fsh.columns = map(str.lower, fsh.columns)
        fsh.columns = [c.replace('__', '_') for c in fsh.columns]
        fsh.replace(['?','None','nan','Nan',' ','NaT','#REF!'],np.nan,inplace=True)
        fsh = trim_all_columns(fsh)
        fsh=fsh.drop_duplicates(keep='last')
        df = pd.DataFrame(fsh)
        return(df)
    
    


def trim_all_columns(df):
        trim_strings = lambda x: x.strip() if isinstance(x, str) else x
        return df.applymap(trim_strings)

def gmode(lst):
    try:
        #print(lst)
        return(lst.mode()[0])
    except:
        next


def call_napercentage(data_train):
    op = pd.DataFrame(data_train.isnull().sum()/data_train.shape[0]*100)
    op = op.reset_index()
    op.rename(columns={'index':'variable_name'},inplace=True)
    op.rename(columns={0:'na%'},inplace=True)
    op=op.sort_values(by='na%',ascending=False)
    return(op)



def get_numcolnames(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numdf = df.select_dtypes(include=numerics)
    cols_nums =numdf.columns
    cols_nums = cols_nums.tolist()
    return(cols_nums)

def get_catcolnames(df):
    categoric = ['object']
    catdf = df.select_dtypes(include=categoric)
    cols_cat = catdf.columns
    cols_cat = cols_cat.tolist()
    return(cols_cat)
    
def concat_columns(temp_df,xlab):
    columns = xlab #=['ncb','age_of_vehicle'] # grouping of 
    columns_conc = ("_".join(columns))
    temp_df[f'{columns_conc}_combi'] = temp_df[columns].astype(str).astype(str).apply('|'.join, axis=1)
    return(temp_df)


def get_quatile_df(data):
    return(pd.DataFrame.quantile(data,[0,0.01,0.02,0.03,0.04,.05,0.25,0.50,0.75,0.85,.95,0.99,0.991,0.992,0.993,0.994,0.995,0.995,0.996,0.997,0.998,0.8999,1]))

def get_quatilebin(data,bin_list):
    return(pd.DataFrame.quantile(data,bin_list))


def do_dataprofiling(data, leveli):
    # data = x
    print('DATA SHAPE', data.shape)
    data = pd.DataFrame(data)
    tempi = pd.DataFrame()
    tempi = pd.DataFrame(data.dtypes, columns=['dtypes'])
    tempi = tempi.reset_index()
    tempi['Name'] = tempi['index']
    tempi = tempi[['Name', 'dtypes']]
    tempi['Missing_Count'] = data.isnull().sum().values
    tempi['Missing_Perct'] = round(tempi['Missing_Count'] / data.shape[0] * 100, 2)
    tempi['Uniques_Count'] = data.nunique().values
    tempi['Uniques_Perct'] = round(tempi['Uniques_Count'] / data.shape[0] * 100, 2)

    tempi['Zeros_count'] = data.isin([0]).sum().tolist()  # data[data == 0].count(axis=0).values
    tempi['Zeros_Perct'] = round(tempi['Zeros_count'] / data.shape[0] * 100, 2)

    tempi['Ones_count'] = data.isin([1]).sum().tolist()  # data[data == 1].count(axis=0).values
    tempi['Ones_Perct'] = round(tempi['Ones_count'] / data.shape[0] * 100, 2)

    tempi['mcp'] = np.nan

    def mode_perC(data, i0):
        # i =  'status_6'
        xi = data[f'{i0}'].value_counts(dropna=False)
        xi = pd.DataFrame(xi)
        xi.reset_index(inplace=True)
        xi.rename(columns={'index': 'colanme', 0: f'{i0}'}, inplace=True)
        xi.sort_values(by=f'{i0}')
        mode_name = xi.iloc[0, 0]
        mode_count = xi.iloc[0, 1]
        mode_perC = round((mode_count / data.shape[0]) * 100, 3)
        m = f'{mode_name}/ {mode_count}/ %{mode_perC}'
        return m

    # Computing MCp
    for i1 in tempi['Name'].unique():
        # print(mode_perC(data,f'{i}'))
        idi = tempi[tempi['Name'] == f'{i1}'].index
        tempi.loc[idi, 'mcp'] = mode_perC(data, f'{i1}')

    # Computing Levels
    leveli = 10
    tempi['Levels'] = 'empty'
    for i2, m in enumerate(tempi['Name']):
        # print(i,m)
        if len(data[f'{m}'].value_counts()) <= leveli:
            # print(data[f'{m}'].value_counts())
            tab = data[f'{m}'].unique()
            tempi.loc[i2, 'Levels'] = f'{tab}'
    tempi['N'] = data.shape[0]

    # Numerical describe func

    num_cols = get_numcolnames(data)

    di = data[num_cols].describe().T
    di.reset_index(inplace=True)
    di.rename(columns={'index': 'Name'}, inplace=True)

    ret_df = pd.merge(tempi, di, on='Name', how='outer')
    ret_df = ret_df.replace(np.nan, '', regex=True)
    ret_df = ret_df.sort_values('Missing_Perct', ascending=False)

    ret_df = ret_df.round(3)
    print('-' * 50)
    print('DATA TYPES:\n', tempi['dtypes'].value_counts(normalize=False))

    a = call_napercentage(data)
    miss_col_great30 = a[a['na%'] > 30].shape[0]
    miss_col_less30 = a[a['na%'] <= 30].shape[0]
    miss_col_equal2_0 = a[a['na%'] == 0].shape[0]
    miss_col_equal2_100 = a[a['na%'] == 100].shape[0]
    print('\nMISSING REPORT:-')
    print('-' * 100,
          '\nTotal Observation                       :', tempi.shape[0],
          '\nNo of Columns with >30% data missing    :', miss_col_great30,
          '\nNo of Columns with <30% data missing    :', miss_col_less30,

          '\nNo of Columns with =0% data missing     :', miss_col_equal2_0,
          '\nNo of Columns with =100% data missing   :', miss_col_equal2_100, '\n', '-' * 100)

    return (ret_df)


def get_firstReport(df,leveli=30):
    temp_df = pd.DataFrame()
    temp_df['actual_colName'] = df.columns
    temp_df['col_index'] = [f'col_{i}' for i in range(0, len(df.columns))]
    temp_df = temp_df[['col_index', 'actual_colName']]

    idict = dict(zip(temp_df['col_index'], temp_df['actual_colName']))
    df.columns = temp_df['col_index']

    date_list = get_partialcol_match(df, 'date')
    # print(date_list)
    cat_list = get_catcolnames(df)
    cat_list = [i for i in cat_list if i not in date_list]

    # print(cat_list)
    def describe1_cat(df, xcol):
        d1 = df[f'{xcol}'].value_counts(dropna=False).reset_index().rename(columns={f'{xcol}': f'{xcol}_count'})
        d2 = df[f'{xcol}'].value_counts(dropna=False, normalize=True).reset_index().rename(
            columns={f'{xcol}': f'{xcol}_percent'})
        d2[f'{xcol}_percent'] = round(d2[f'{xcol}_percent'] * 100, 2)
        d3 = pd.merge(d1, d2, on='index')
        d3.sort_values(by=[f'{xcol}_percent'], ascending=False)
        return (d3)

    i1 = []
    for i in df.columns:
        # print(i)
        if len(i) >= 31:
            i1.append(i[:31])
        else:
            i1.append(i)
    # print(i1)
    df.columns = i1

    temp_list = [];
    err = []
    for i in cat_list:
        # print(i)
        try:
            txt = f"{i}=describe1_cat(df,'{i}')";
            exec(txt)
            txt = f"temp_list.append('{i}')";
            exec(txt)
        except:
            err.append(i)
    print('ERROR COLUMNS ARE:', err)
    num_list = get_numcolnames(df)
    num_list = [i for i in num_list if i not in date_list]

    analysis_num = df[num_list].describe().T
    analysis_num = analysis_num.reset_index()
    analysis_num = pd.DataFrame(analysis_num);
    temp_list.append(f'analysis_num')
    # analysis_num.to_excel('temp_data/analysis_num.xlsx',index=True)

    profiling = do_dataprofiling(df, leveli);
    profiling['colname'] =  profiling['Name'].map(idict)
    temp_list.append(f'profiling')
    # re.to_excel('temp_data/analysis_profiling.xlsx',index=False)

    temp_list.insert(0, temp_list.pop())
    temp_list.insert(1, temp_list.pop())
    # Create some Pandas dataframes from some data.
    df_list = temp_list  # ['df1','df2','df3']

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter('temp_data/Descriptive_Report.xlsx', engine='xlsxwriter')

    for i in df_list:
        run_cmd = f"{i}.to_excel(writer, sheet_name='{i}',index=False)";
        exec(run_cmd)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()

    print('completed ...........................!!')
    return (profiling)

# 2. VISUALIZATION FUNCTION :




