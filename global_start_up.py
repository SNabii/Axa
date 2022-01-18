#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# SETTING BASIC ENVIORMENT :

# LOADING  BASIC LIBRARY ---

import qgrid
import os
import pandas as pd 
import numpy as np

import scipy.stats as stats
import qgrid

from scipy.stats import norm, skew, kurtosis, boxcox #for some statistics
# import pyarrow.parquet as pq
# import pyarrow as pa
from scipy import interp

# from dplython import select, DplyFrame, X, arrange, count, sift, head, summarize, group_by, tail, mutate

from sklearn.model_selection import validation_curve
import sklearn.metrics as metrics

from sklearn.metrics import auc, accuracy_score
from sklearn.metrics  import plot_roc_curve
from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold
#from catboost import CatBoostClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from itertools import product, chain
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Library visualisation ---
import seaborn as sns
import matplotlib.pyplot as plt
# for animated visualizations
from bubbly.bubbly import bubbleplot
import plotly_express as px
import plotly
import plotly.offline as py

def get_stime():
    import time
    stime = time.time()
    return(stime)

from datetime import datetime
t_stamp = datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p")

def get_valcount(temp,xcol,na,norm):
    return(gview(temp[f'{xcol}'].value_counts(dropna=norm,normalize=na)))

def ret_analysis(df,xindex,xtarget):
    #xindex = ['product_tot','sum_insured_tot']
    #xtarget = ['target']
    combi =  xindex + xtarget
    re = df.groupby(combi).size().reset_index().rename(columns={0:'count'}).pivot_table(index=xindex,columns=xtarget)
    re =  rename_multiindex(re)
    re = pd.DataFrame(re)
    re =re.replace(np.nan,0)
    re['tot'] = re['count_1.0'] + re['count_0.0']
    re['ret%'] = round(re['count_1.0']/re['tot']*100,2)
    
    return(re)

def reg_plot(df,col_name,orde):
    df[f'{col_name}bin']=pd.cut(df[f'{col_name}'],bins=np.linspace(0, 100, 11))
    t1=df.pivot_table(index=f'{col_name}bin', columns='target', aggfunc='size')
    t1 = t1.reset_index()
    t1.rename(columns={0.0:'churn',1.0:'renew'},inplace=True)
    t1['tot'] =  t1['churn']+t1['renew']
    t1['renew_rate'] = round((t1['renew']/t1['tot'])*100,2)
    t1=t1.replace(np.nan,0)
    #t1 = t1[:-2]
    sns.regplot(np.linspace(t1[f'{col_name}bin'][0].left, t1[f'{col_name}bin'][(t1.shape[0]-1)].right, t1.shape[0]),t1['renew_rate'],order=orde)
    
    
def get_binplotting(dfx,colx):
    dfx[f'{colx}'] = dfx[f'{colx}'].astype('float64')
    nbins = get_quatilebin(dfx[[f'{colx}','age_1']],[0,0.25,0.50,0.75,0.85,0.95,1.0])
    dfx[f'{colx}_bin']= pd.cut(dfx[f'{colx}'],bins=nbins[f'{colx}'].unique().tolist())
    get_plot_ordVs_target(dfx,cat_col=f'{colx}_bin',y_lab='target',Top_n=20,thres=np.nan,ytag=1,prt='N')


def call_lfiles(path,ftype):
    import os
    import glob

    path = path#'D:/data/Analytics Track/Customer Retention Motor/Input_data2/csv/'
    extension = ftype#'csv'
    os.chdir(path)
    data_file = glob.glob('*.{}'.format(extension))
    return(data_file)

def plot_crossTab(temp_df,x_vari):
    temp=pd.crosstab(temp_df[f'{x_vari}'],temp_df['target'])
    temp.plot(kind='bar', stacked=True, grid=True, figsize=(15, 7), )
    temp["Ratio"] =  temp[1] / (temp[0]+temp[1])
    print(temp)
    return(temp)

from scipy.stats import skew, kurtosis
def get_skewkurt(data,icolname):
    s= round(skew(data[f'{icolname}']),3)
    k= round(kurtosis(data[f'{icolname}']),3)
    return(s,k)
    
def get_duplicate_columns(temp):
    er = pd.DataFrame(columns = ['Col_nam'])
    er['Col_nam'] = temp.columns
    er=er.groupby('Col_nam')['Col_nam'].agg(['count']).sort_values(['count'],ascending=False)
    er.reset_index()
    return(er)
    
def rename_multiindex(qr):
    level_one = qr.columns.get_level_values(0).astype(str)
    level_two = qr.columns.get_level_values(1).astype(str)
    qr.columns = level_one +'_'+ level_two
    return(qr)


from string import punctuation
def do_columnclean_withOutNum(lst):
    if lst is not None:
        si = lst # 'Die sel 
        si = str(lst)
#         si = si.strip()
#         si=si.strip(punctuation)
#         si = si.strip()
        si = ''.join([i for i in si if not i.isdigit()])
        si = si.lower()       
        return(si)
    
def do_columnclean_withNum(lst):
    if lst is not None:
        si = lst # 'Die sel '
        si = str(lst)
        si = si.strip()
        si=si.strip(punctuation)
        si = si.strip()
        #si = ''.join([i for i in si if not i.isdigit()])
        si = si.lower() 
        si = si.replace(' ','_')
        return(si)
    
def do_lower_case():
    if lst is not None:
        si = lst # 'Die sel '
        si = si.strip()
        si = si.lower()       
        return(si)
    
def reduce_catgory(data,coli,breaki,roundi,target):
    # coli =  'retail_city'
    # breaki = 5
    # roundi = 1
    
    df2 = data 

    try:
        vi = df2.groupby([f'{coli}',f'{target}']).size().reset_index().pivot(index=f'{coli}', columns=f'{target}', values=0)
    except:
        vi = pd.crosstab(temp_df[f'{coli}'], temp_df[f'{target}'])
    #vi.rename(columns={0:'no',1:'yes'},inplace=True)
    vi = vi.reset_index()
    vi.columns=[f'{coli}','no','yes']
    vi.fillna(0,inplace=True) 
    vi['total'] =  vi['no']+vi['yes']

    vi['no_score'] =  round((vi['no']/vi['total']),4)
    vi['yes_score'] =  round((vi['yes']/vi['total']),4)

    #vi['total_qcut'] = pd.qcut(vi['yes_score'], breaki)
    vi['yes_score']  = round(vi['yes_score'],roundi)
    #vi.sort_values(by='total_qcut')

    from itertools import groupby
    N = vi['yes_score'].tolist()
    slist = [list(j) for i, j in groupby(N)]
    #print(slist)

    ti=len(slist)

    temp_list = []
    for i in slist:
        #print(i[0])
        temp_list.append(i[0])

    temp_list=set(temp_list)
    temp_list =list(temp_list)

    temp_list = temp_list

    #len(set(temp_list))

    tr = pd.DataFrame()
    tr['cut_val'] = temp_list
    tr['cut_val']=tr['cut_val'].astype('float')
    tr = tr.sort_values(by='cut_val',ascending=False)



    collecti = []
    for i in range(1,(tr.shape[0]+1)):
        collecti.append(f'w{i}')


    tr['level'] = collecti
    #print(tr)

    area_dict = dict(zip(tr['cut_val'], tr['level']))
    #print(area_dict)

    vi[f'{coli}_level'] = vi['yes_score'].map(area_dict)
    vi.sort_values(by='yes_score',ascending=False,inplace=True)
#     vt = temp_riz.groupby(['retail_city_level']).agg({'yes_score':['count','min','max']})
#     vt = vt.reset_index()
#     vt.columns =  ['yes_score','count','min','max']
#     vt.sort_values(by='max',ascending=False)

    return(vi) 
    
def plot_stack_bar2catvariables(temp_df,x_var,y_var):
    plt.figure(figsize=(16,9))
    sns.countplot(hue=f'{y_var}',x =f'{x_var}',data = temp_df, palette="gist_ncar",dodge=False)
    plt.title(f'{x_var} - {y_var}')
    plt.show()
    
# b is a binding data
# q1 is a extract data
def getlent(lst):
    lst = str(lst)
    return(len(lst))
def rm_l0(lst):
    lst =  str(lst)
    if len(lst)==10:
        lst = lst.lstrip('0')
        return(lst)
    else:
        return(lst)
def rm_l1(lst):
    lst =  str(lst)
    if len(lst)==11:
        lst = lst.lstrip('0')
        return(lst)
    else:
        return(lst)

# All Func ---

def seq(start, stop, step=1):
    n = int(round((stop - start)/float(step)))
    if n > 1:
        return([start + step*i for i in range(n+1)])
    elif n == 1:
        return([start])
    else:
        return([])
    
from dateutil import parser
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

def gview(data):
    print('DIMENSION',data.shape)
    return(qgrid.show_grid(data,show_toolbar=True,grid_options={'forceFitColumns': False,'highlightSelectedCell': True,
        'highlightSelectedRow': True}))

def trim_all_columns(df):
        trim_strings = lambda x: x.strip() if isinstance(x, str) else x
        return df.applymap(trim_strings)

def gmode(lst):
    try:
        #print(lst)
        return(lst.mode()[0])
    except:
        next

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


def get_partialcol_match(final_df,txt):
    date_colist = final_df[final_df.columns[final_df.columns.to_series().str.contains(f'{txt}')]].columns
    date_colist = date_colist.tolist()
    return(date_colist)  
    
    
def summarise_yourdf(df,leveli):
    print(f"Dataset Shape: {df.shape}")    

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing_Count'] = df.isnull().sum().values
    summary['Missing_Perct'] = round(summary['Missing_Count']/df.shape[0]*100,2)
    summary['Uniques_Count'] = df.nunique().values
    summary['Uniques_Perct'] = round(summary['Uniques_Count']/df.shape[0]*100,2)
    
    #summary['First Value'] = df.loc[0].values
    #summary['Second Value'] = df.loc[1].values
    #summary['Third Value'] = df.loc[2].values
    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 
    summary['Zeros_count'] = df[df == 0].count(axis=0).values
    summary['Zeros_Perct'] = round(summary['Zeros_count']/df.shape[0]*100,2)
    
    
    summary['Levels']= 'empty'
    for i, m in enumerate (summary['Name']):
            #print(i,m)
            if len(df[f'{m}'].value_counts()) <= leveli:
                #print(df[f'{m}'].value_counts())
                tab = df[f'{m}'].unique()
                summary.ix[i,'Levels']=f'{tab}'
    summary['N'] = df.shape[0]
    
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numdf = df.select_dtypes(include=numerics)
    cols_nums =numdf.columns
    categoric = ['object']
    catdf = df.select_dtypes(include=categoric)
    cols_cat = catdf.columns#names of all the columns
    

    desc=df[cols_nums].describe().T
    desc = desc.reset_index()
    desc = pd.DataFrame(desc)
    desc.rename(columns={f'{desc.columns[0]}':'Name'}, inplace=True)
    desc.drop(['count'], axis=1,inplace=True)
    desc = round(desc,2)
   # desc

    merged_inner=pd.merge(summary, desc, on='Name', how='outer')
    merged_inner = merged_inner.replace(np.nan, '', regex=True)
    merged_inner = merged_inner.sort_values('Missing_Perct',ascending=False)
    merged_inner.to_excel('profiling.xlsx',index=False)
    return (merged_inner)
    
    


def do_dataprofiling(data,leveli):
    #data = x
    print('DATA SHAPE',data.shape)
    data = pd.DataFrame(data)
    tempi = pd.DataFrame()
    tempi = pd.DataFrame(data.dtypes,columns=['dtypes'])
    tempi = tempi.reset_index()
    tempi['Name'] = tempi['index']
    tempi = tempi[['Name','dtypes']]
    tempi['Missing_Count'] = data.isnull().sum().values
    tempi['Missing_Perct'] = round(tempi['Missing_Count']/data.shape[0]*100,2)
    tempi['Uniques_Count'] = data.nunique().values
    tempi['Uniques_Perct'] = round(tempi['Uniques_Count']/data.shape[0]*100,2)

    tempi['Zeros_count'] = data.isin([0]).sum().tolist()#data[data == 0].count(axis=0).values
    tempi['Zeros_Perct'] = round(tempi['Zeros_count']/data.shape[0]*100,2)

    tempi['Ones_count'] = data.isin([1]).sum().tolist()#data[data == 1].count(axis=0).values
    tempi['Ones_Perct'] = round(tempi['Ones_count']/data.shape[0]*100,2)

    tempi['mcp'] = np.nan

    def mode_perC(data,i0):
        #i =  'status_6'
        xi = data[f'{i0}'].value_counts(dropna=False)
        xi = pd.DataFrame(xi)
        xi.reset_index(inplace=True)
        xi.rename(columns= {'index':'colanme',0:f'{i0}'},inplace=True)
        xi.sort_values(by=f'{i0}')
        mode_name = xi.iloc[0,0]
        mode_count = xi.iloc[0,1]
        mode_perC = round((mode_count/data.shape[0])*100,3)
        m = f'{mode_name}/ {mode_count}/ %{mode_perC}'
        return m    


    # Computing MCp
    for i1 in tempi['Name'].unique():
        #print(mode_perC(data,f'{i}'))
        idi = tempi[tempi['Name'] == f'{i1}'].index
        tempi.loc[idi,'mcp'] = mode_perC(data,f'{i1}')

    # Computing Levels
    leveli=10    
    tempi['Levels']= 'empty'
    for i2, m in enumerate (tempi['Name']):
            #print(i,m)
            if len(data[f'{m}'].value_counts()) <= leveli:
                #print(data[f'{m}'].value_counts())
                tab = data[f'{m}'].unique()
                tempi.loc[i2,'Levels']=f'{tab}'
    tempi['N'] = data.shape[0]

    # Numerical describe func

    num_cols =get_numcolnames(data)


    di =data[num_cols].describe().T
    di.reset_index(inplace=True)
    di.rename(columns={'index':'Name'},inplace=True)


    ret_df =pd.merge(tempi, di, on='Name', how='outer')
    ret_df = ret_df.replace(np.nan, '', regex=True)
    ret_df = ret_df.sort_values('Missing_Perct',ascending=False)


    ret_df = ret_df.round(3)
    print('-'*50)
    print('DATA TYPES:\n',tempi['dtypes'].value_counts(normalize=False))

    a = call_napercentage(data)
    miss_col_great30 = a[a['na%']>30].shape[0]
    miss_col_less30 = a[a['na%'] <=30].shape[0]
    miss_col_equal2_0 = a[a['na%'] ==0].shape[0]
    miss_col_equal2_100 = a[a['na%'] ==100].shape[0]
    print('\nMISSING REPORT:-')
    print('-'*100,
           '\nTotal Observation                       :',tempi.shape[0],
          '\nNo of Columns with >30% data missing    :',miss_col_great30,
          '\nNo of Columns with <30% data missing    :',miss_col_less30,

          '\nNo of Columns with =0% data missing     :',miss_col_equal2_0,
         '\nNo of Columns with =100% data missing   :',miss_col_equal2_100,'\n','-'*100)

    return(ret_df)

# outlier treatment ----

def get_quatile_df(data):
    return(pd.DataFrame.quantile(data,[0,0.01,0.02,0.03,0.04,.05,0.25,0.50,0.75,0.85,.95,0.99,0.991,0.992,0.993,0.994,0.995,0.995,0.996,0.997,0.998,0.8999,1]))

def get_quatilebin(data,bin_list):
    return(pd.DataFrame.quantile(data,bin_list))

def get_box_plots(data):
    for coli in get_numcolnames(data):
        plt.figure(figsize=(15,9))
        plt.boxplot(data[f'{coli}'],0,'gD')
        plt.title(f'{coli}')
        plt.show()
        
def get_outliers(df,l_band , u_band):
    Q1 = df.quantile(l_band)
    Q3 = df.quantile(u_band)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return(df)
#outliers function
def get_median_filtered(signal, threshold=3):
    signal = signal.copy()
    difference = np.abs(signal - np.median(signal))
    median_difference = np.median(difference)
    if median_difference == 0:
        s = 0
    else:
        s = difference / float(median_difference)
    mask = s > threshold
    signal[mask] = np.median(signal)
    return signal 


def get_binary_distribution(data,target):
    f,ax=plt.subplots(1,2,figsize=(15,6))
    data[f'{target}'].value_counts().plot.pie(explode=[0,1],autopct='%1.1f%%',ax=ax[0],shadow=True,colors = ['red','green'])
    ax[0].set_title(f'{target}')
    ax[0].set_ylabel('')
    sns.countplot(f'{target}',data=data,ax=ax[1],palette= ['red','green'])
    ax[1].set_title(f'{target}')
    plt.show()
    
def plx_histgroupby(df,target,numcol):
    plt.figure(figsize=(16,7))

    sns.distplot(df[df[f'{target}'] == 1][f'{numcol}'], color='g', label = "PROFIT") 
    sns.distplot(df[df[f'{target}'] == 0][f'{numcol}'], color='r', label = "LOSS") 

    plt.xlabel(f"{numcol} Values")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {numcol} Mean for profit and loss ", fontsize=14)
    plt.legend()

    plt.show()    
    
def plx_histograme(df,colx,binz=20):
    #colx = 'pnl'
    x=df[f'{colx}']
    plt.figure(figsize=(15,7))
    result = plt.hist(x, bins=binz, color='c', edgecolor='k', alpha=0.65)
    plt.axvline(x.mean(), color='k', linestyle='dashed', linewidth=1)
    plt.axvline(x.median(), color='k', linestyle='dashed', linewidth=1)
    min_ylim, max_ylim = plt.ylim()
    plt.text(x.mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(x.mean()),)
    plt.text(x.median()*1.1, max_ylim*0.8, 'Median: {:.2f}'.format(x.median()))
    plt.show()

def get_plot_catVs_target(df,cat_col,y_lab,Top_n,thres,ytag,prt):
    i = cat_col#"Age_Of_Vehicle"
    y2 = y_lab#"Renewed2"
    Top_n = Top_n #15
    ytag = ytag
    col_count  = df[f'{i}'].value_counts()
    #print(col_count)
    col_count = col_count[:Top_n,]

    col_count1 = df[f'{i}'].value_counts(normalize=True)*100
    col_count1 = col_count1[:Top_n,]
    vol_inperc = col_count1.sum()
    vol_inperc = round(vol_inperc,2)

    tmp = pd.crosstab(df[f'{i}'], df[f'{y2}'], normalize='index') * 100
    tmp = pd.merge(col_count, tmp, left_index=True, right_index=True)
    tmp.rename(columns={0:'NotRenwed%', 1:'Renewed%'}, inplace=True)
    if 'NotRenwed%' not in tmp.columns:
        print("NotRenwed% is not present in ",i)
        tmp['NotRenwed%'] = 0
    if 'Renewed%' not in tmp.columns:
        print("Renewed% is not present in ",i)
        tmp['Renewed%'] = 0
    
    tmp1 = pd.crosstab(df[f'{i}'], df[f'{y2}'])
    tmp1.rename(columns={0:'NR_count', 1:'R_count'}, inplace=True)
    if 'NR_count' not in tmp1.columns:
        print("NR_count is not present in ",i)
        tmp1['NR_count'] = 0
    if 'R_count' not in tmp1.columns:
        print("R_count is not present in ",i)
        tmp1['R_count'] = 0

    tmpz=pd.merge(tmp,tmp1,
        left_index=True,
        right_index=True)
    tmpz['Tot'] = tmpz['NR_count'] + tmpz['R_count'] 
    tmpz['Renewed%'] =  round(tmpz['Renewed%'],2)
    tmpz['Mean'] = (tmpz['R_count']/tmpz['Tot'])*100
    tmpz['Nperformer'] = np.where(tmpz['Renewed%'] < tmpz['Mean'] ,1,0)
    #tmpz.sort_index(inplace=True)
    
    tmpz.fillna(0)
    if prt == 'Y':
        gview(tmpz)
        tmpzi = tmpz.reset_index()  
        #tmpzii = tmpzi .join(pd.DataFrame(tmpzi.index.str.split('-').tolist()))
        #tmpzi = pd.concat([tmpzi,DataFrame(tmpzi.index.tolist())], axis=1, join='outer')
        #tmpzi.to_excel("tmpz.xlsx")
      
   
    plt.figure(figsize=(16,7))
    g=sns.barplot(tmpz.index, tmpz[f'{i}'], alpha=0.8,order = col_count.index)
    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.title(f'{i} with {vol_inperc}%', fontsize = 16,color='blue')
    #g.set_title(f'{i}')
    plt.ylabel('Volume', fontsize=12)
    plt.xlabel(f'{i}', fontsize=12)
    plt.xticks(rotation=90)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{}\n{:1.2f}%'.format(round(height),height/len(df)*100),
            ha="center", fontsize=12, color='blue')

    gt = g.twinx()

    if ytag == 1:
        values = tmpz['Renewed%'].values # <--- store the values in a variable for easy access
        gt = sns.pointplot(x=tmpz.index, y='Renewed%', data=tmpz, color='green', legend=True,order=tmpz.index)
    if thres is np.nan:
        gt.set_ylim(0,100)
   

    if ytag == 0:
        values = tmpz['NotRenwed%'].values # <--- store the values in a variable for easy access
        gt = sns.pointplot(x=tmpz.index, y='NotRenwed%', data=tmpz, color='red', legend=True,order=tmpz.index)
        #gt.set_ylim(tmp['NotRenwed%'].min()-1,tmp['NotRenwed%'].max()+5)
    if thres is np.nan:
        gt.set_ylim(0,100)
        
        
        
    if thres is np.nan and ytag ==1: 
        gt.axhline(y=(tmpz['R_count'].sum()/tmpz['Tot'].sum())*100, xmax=7, color='blue', linestyle='--')
    else:
        gt.axhline(y=thres, xmin=0, xmax=7, color='blue', linestyle='--')
        
    if thres is np.nan and ytag ==0: 
        gt.axhline(y=(tmpz['NR_count'].sum()/tmpz['Tot'].sum())*100, xmax=7, color='blue', linestyle='--')
    else:
        gt.axhline(y=thres, xmin=0, xmax=7, color='blue', linestyle='--')
        
    #values = tmpz['Renewed%'].values # <--- store the values in a variable for easy access
    j=0
    for c in gt.collections:
        for i, of in zip(range(len(c.get_offsets())), c.get_offsets()):
            gt.annotate(values[j], of, color='brown', fontsize=12, rotation=45)
            j += 1
    plt.show()
    
    
def get_plot_ordVs_target(df,cat_col,y_lab,Top_n,thres,ytag,prt):
    i = cat_col#"Age_Of_Vehicle"
    y2 = y_lab#"Renewed2"
    Top_n = Top_n #15
    ytag = ytag
    
    #df[f'{i}'] = df[f'{i}'].astype(int)
    col_count  = df[f'{i}'].value_counts()
    #print(col_count)
    col_count.sort_index(inplace=True)
    col_count = col_count[:Top_n,]

    col_count1 = df[f'{i}'].value_counts(normalize=True)*100
    col_count1 = col_count1[:Top_n,]
    vol_inperc = col_count1.sum()
    vol_inperc = round(vol_inperc,2)

    tmp = pd.crosstab(df[f'{i}'], df[f'{y2}'], normalize='index') * 100
    tmp = pd.merge(col_count, tmp, left_index=True, right_index=True)
    tmp.rename(columns={0:'NotRenwed%', 1:'Renewed%'}, inplace=True)
    if 'NotRenwed%' not in tmp.columns:
        print("NotRenwed% is not present in ",i)
        tmp['NotRenwed%'] = 0
    if 'Renewed%' not in tmp.columns:
        print("Renewed% is not present in ",i)
        tmp['Renewed%'] = 0

    tmp1 = pd.crosstab(df[f'{i}'], df[f'{y2}'])
    tmp1.rename(columns={0:'NR_count', 1:'R_count'}, inplace=True)
    if 'NR_count' not in tmp1.columns:
        print("NR_count is not present in ",i)
        tmp1['NR_count'] = 0
    if 'R_count' not in tmp1.columns:
        print("R_count is not present in ",i)
        tmp1['R_count'] = 0

    tmpz=pd.merge(tmp,tmp1,
        left_index=True,
        right_index=True)
    tmpz['Tot'] = tmpz['NR_count'] + tmpz['R_count'] 
    tmpz['Renewed%'] =  round(tmpz['Renewed%'],2)
    tmpz['Mean'] = tmpz['Renewed%'].mean()
    tmpz['Nperformer'] = np.where(tmpz['Renewed%'] < tmpz['Mean'] ,1,0)
    #tmpz.sort_index(inplace=True)
    tmpz.fillna(0)
    
    if prt == 'Y':
        print(tmpz)
        tmpzi = tmpz.reset_index()  
        #tmpzii = tmpzi .join(pd.DataFrame(tmpzi.index.str.split('-').tolist()))
        #tmpzi = pd.concat([tmpzi,DataFrame(tmpzi.index.tolist())], axis=1, join='outer')
        #tmpzi.to_excel("tmpz.xlsx")
      
   
    plt.figure(figsize=(16,7))
    g=sns.barplot(tmpz.index, tmpz[f'{i}'], alpha=0.8)
    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.title(f'{i} with {vol_inperc}%', fontsize = 16,color='blue')
    #g.set_title(f'{i}')
    plt.ylabel('Volume', fontsize=12)
    plt.xlabel(f'{i}', fontsize=12)
    plt.xticks(rotation=90)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{}\n{:1.2f}%'.format(round(height),height/len(df)*100),
            ha="center", fontsize=12, color='blue')

    gt = g.twinx()

    if ytag == 1:
        values = tmpz['Renewed%'].values # <--- store the values in a variable for easy access
        gt = sns.pointplot(x=tmpz.index, y='Renewed%', data=tmpz, color='green', legend=True)
    
    if ytag == 0:
        values = tmpz['NotRenwed%'].values # <--- store the values in a variable for easy access
        gt = sns.pointplot(x=tmpz.index, y='NotRenwed%', data=tmpz, color='red', legend=True)
        gt.set_ylim(tmp['NotRenwed%'].min()-1,tmp['NotRenwed%'].max()+5)
        
        
    if thres is np.nan:
        gt.set_ylim(0,100)
        gt.axhline(y=(tmpz['R_count'].sum()/tmpz['Tot'].sum())*100, xmax=7, color='blue', linestyle='--')
    else:
        gt.set_ylim(0,100)
        gt.axhline(y=thres, xmin=0, xmax=7, color='blue', linestyle='--')
   


        
        
    if thres is np.nan and ytag ==1: 
        gt.axhline(y=(tmpz['R_count'].sum()/tmpz['Tot'].sum())*100, xmax=7, color='blue', linestyle='--')
    else:
        gt.axhline(y=thres, xmin=0, xmax=7, color='blue', linestyle='--')
        
    if thres is np.nan and ytag ==0: 
        gt.axhline(y=(tmpz['NR_count'].sum()/tmpz['Tot'].sum())*100, xmax=7, color='blue', linestyle='--')
    else:
        gt.axhline(y=thres, xmin=0, xmax=7, color='blue', linestyle='--')
        
    #values = tmpz['Renewed%'].values # <--- store the values in a variable for easy access
    j=0
    for c in gt.collections:
        for i, of in zip(range(len(c.get_offsets())), c.get_offsets()):
            gt.annotate(values[j], of, color='brown', fontsize=12, rotation=45)
            j += 1
    plt.show()
    
from statsmodels.stats.outliers_influence import variance_inflation_factor    

def calculate_vif_(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

# Confusion Matrix function

def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, cmap="YlGnBu", xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True, annot_kws={'size':50})
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

# This function plots the confusion matrices given y_i, y_i_hat.
def plot_confusion_matrix_pr(test_y, predict_y,lab_list):
    from sklearn.metrics import confusion_matrix
    C = confusion_matrix(test_y, predict_y)

    
    A =(((C.T)/(C.sum(axis=1))).T)
    B =(C/C.sum(axis=0))
    labels = lab_list# [1, 2, 3]#class_count#[1,2,3,4,5,6,7,8,9]
    # representing A in heatmap format
    print("-"*20, "Confusion matrix", "-"*20)
    plt.figure(figsize=(20,7))
    sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()

    print("-"*20, "Precision matrix (Columm Sum=1)", "-"*20)
    plt.figure(figsize=(20,7))
    sns.heatmap(B, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()
    
    # representing B in heatmap format
    print("-"*20, "Recall matrix (Row sum=1)", "-"*20)
    plt.figure(figsize=(20,7))
    sns.heatmap(A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()

def get_Modelreport(model,x_train,y_train,x_test,y_test,Model_name):
    print(f'***************Report for Model :{Model_name}***************\n\n')
    lab_list =  y_train.unique().tolist()
    y_pred_rf = model.predict(x_test)

    print("Training Accuracy: ", model.score(x_train, y_train))
    print('Testing Accuarcy: ', model.score(x_test, y_test))

    # making a classification report
    cr = classification_report(y_test,  y_pred_rf)
    print(cr)
    
    # making a confusion matrix
    plot_confusion_matrix_pr(y_test, y_pred_rf,lab_list)   
    
    
    
# Plot scatter plot with target categorical variables --
def plot_scatter_withtarget(data,x_lab,y_lab,target):
    try:
        fig = px.scatter(data, x = f'{x_lab}', y = f'{y_lab}', color = f'{target}',
                        marginal_x = 'rug', marginal_y = 'histogram')
        fig.show()
    except:
        print('parameter is missing')
        
def plot_animationbubble(data,x_lab,y_lab,target,time_col,size_col):
    figure = bubbleplot(dataset = data, x_column = f'{x_lab}', y_column = f'{y_lab}', 
        bubble_column = f'{target}', time_column = f'{time_col}', size_column = f'{size_col}', color_column = f'{target}', 
        x_title = f"{x_lab}", y_title = f"{y_lab}", title = f'{x_lab} vs {y_lab} vs {size_col} as Customer',
        x_logscale = False, scale_bubble = 3, height = 650)

    py.iplot(figure, config={'scrollzoom': True})
    
# Distribution of target feature
def plot_densityqq(y,plottitle,plotname):
	from scipy.stats import norm, skew, probplot #for some statistics
	plt.figure(0,figsize=[15,5])
	plt.subplots_adjust(wspace=0.2, hspace=0.5)
	plt.subplot(1,2,1)
	(mu, sigma) = norm.fit(y)
	#print( 'mu = {:.2f} and sigma = {:.2f}'.format(mu, sigma))
	sns.distplot(y, fit=norm)
	plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
	            loc='best')
	plt.ylabel('Frequency')
	plt.title(plottitle)
	# QQ-plot wrt normal distribution
	plt.subplot(1,2,2)
	res = probplot(y, plot=plt)
	#plt.savefig('figures/'+plotname+'.pdf',bbox_inches='tight',transparent=True)
	plt.show()
	plt.close(0)
    
def plot_corrplot(data):
    # calculate the correlation matrix
    corr = data.corr()
    cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)

    def magnify():
        return [dict(selector="th",
                     props=[("font-size", "7pt")]),
                dict(selector="td",
                     props=[('padding', "0em 0em")]),
                dict(selector="th:hover",
                     props=[("font-size", "12pt")]),
                dict(selector="tr:hover td:hover",
                     props=[('max-width', '200px'),
                            ('font-size', '12pt')])]
    

    f=corr.style.background_gradient(cmap, axis=1)\
        .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
        .set_caption("Hover to magify")\
        .set_precision(2)\
        .set_table_styles(magnify())
    display(f)
    
def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(5,5)):
    
    from sklearn.metrics import confusion_matrix
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=None)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    
    
#     for i in range(nrows):
#         for j in range(ncols):
#             c = cm[i, j]
#             p = cm_perc[i, j]
#             if i == j:
#                 s = cm_sum[i]
#                 annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
#             elif c == 0:
#                 annot[i, j] = ''
#             else:
#                 annot[i, j] = '%.1f%%\n%d' % (p, c)
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
#                 annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                #annot[i, j] = '%.1f%%\n%d' % (p, c)
                #annot[i, j] = '%d\n%d' % (c,T)
                annot[i, j] = '%d' % (c)
            elif c == 0:
                annot[i, j] = ''
            else:
                #annot[i, j] = '%.1f%%\n%d' % (p, c)
                #annot[i, j] = '%d\n%d' % (c,T)
                annot[i, j] = '%d' % (c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
from sklearn.metrics import confusion_matrix
# This function plots the confusion matrices given y_i, y_i_hat.
def plot_cm_pc(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)

    
    A =(((C.T)/(C.sum(axis=1))).T)
    B =(C/C.sum(axis=0))
    labels = [0,1]#[1, 2, 3, 4, 5]#class_count#[1,2,3,4,5,6,7,8,9]
    # representing A in heatmap format
    print("-"*20, "Confusion matrix", "-"*20)
    plt.figure(figsize=(20,7))
    sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()

    print("-"*20, "Precision matrix (Columm Sum=1)", "-"*20)
    plt.figure(figsize=(20,7))
    sns.heatmap(B, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()
    
    # representing B in heatmap format
    print("-"*20, "Recall matrix (Row sum=1)", "-"*20)
    plt.figure(figsize=(20,7))
    sns.heatmap(A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()   

    
def get_auc_score(model,x_df,y_df):
    from sklearn.metrics import roc_curve, auc #for model evaluation
    y_pred_quant = model.predict(x_df)
    fpr, tpr, thresholds = roc_curve(y_df, y_pred_quant)
    return(auc(fpr, tpr))


def get_auc_plot(model,x_df,y_df):
    from sklearn.metrics import roc_curve, auc #for model evaluation
    y_pred_quant = model.predict(x_df)
    fpr, tpr, thresholds = roc_curve(y_df, y_pred_quant)
    print(auc(fpr, tpr))
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve for Retention Model')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)

def plot_precision_recall_vs_threshold(model,x_tr,y_tr, title):    
    from sklearn.metrics import precision_recall_curve    
    probablity = model.predict_proba(x_tr)[:, 1]    
    plt.figure(figsize = (18, 5))    
    precision, recall, threshold = precision_recall_curve(y_tr, probablity)    
    plt.plot(threshold, precision[:-1], 'b-', label = 'precision', lw = 3.7)    
    plt.plot(threshold, recall[:-1], 'g', label = 'recall', lw = 3.7)    
    plt.xlabel('Threshold')    
    plt.legend(loc = 'best')    
    plt.ylim([0, 1])    
    plt.title(title)    
    plt.show() 