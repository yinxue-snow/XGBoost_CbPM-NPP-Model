# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 11:14:32 2023

@author: Zhangyinxue
"""

"""
基于2022年全球月平均数据和XGBoost_CbPM模型，计算每个月的NPP
"""


import os
import xarray as xr
import numpy as np
import pickle
import netCDF4 as nc
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
import sys
import seaborn as sns
from sklearn.model_selection import GridSearchCV
import joblib
import random
from scipy.stats import gaussian_kde
 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
from sklearn.metrics import explained_variance_score, r2_score, mean_absolute_error, mean_squared_error as MSE
import xgboost
from xgboost import XGBRegressor, plot_importance
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from time import time
import datetime
from sklearn.model_selection import KFold, cross_val_score as CVS
import joblib
from sklearn.preprocessing import MinMaxScaler
import pickle

datapath=r'F:\01NPPpaper\JGROcean\审稿过程\一审回复\补充内容\github\dataset_profile.xlsx'
data = pd.read_excel(datapath, sheet_name='XGBoost2',index_col=0) 
print(data.head())
print(data.shape)
index = data.index
col = data.columns

data_train, tmp = train_test_split(data, test_size=0.2, random_state=0)
tt=14
X_train = data_train.iloc[:, 0:tt]
y_train = data_train.iloc[:, tt:]
x_tmp=tmp.iloc[:, 0:tt]
y_tmp=tmp.iloc[:, tt:]
model = xgboost.XGBRegressor(n_estimators=300,max_depth=5,learning_rate=0.01, random_state=420)        
xgbr = MultiOutputRegressor(model).fit(X_train, y_train)

# load input
path=r'F:\01NPPpaper\JGROcean\审稿过程\一审回复\补充内容\github\XGBoost_CbPM_input'
file=os.listdir(path)
for i in range(1,len(file)):
    folder = os.path.join(path, file[i-1])
    tmpt=xr.open_dataset(folder)
    para = np.array(tmpt['modelinput'])
    del tmpt
    temp2 = np.transpose(para)
    npp=xgbr.predict(temp2)           
    # create .nc
    savepath = r'F:\NPP\08 Global application\npp_python_output'
    datafile = nc.Dataset(savepath + "\\"  + "NPP"  +folder[-9:-3]+'.nc', 'w')
    sample = datafile.createDimension('sample', len(temp2))
    depth = datafile.createDimension('depth', 32)
    nppdata = datafile.createVariable('nppdata', 'f4', ( 'sample', 'depth'))
    nppdata.units = "mg C m$^{\mathregular{-2}}$ d$^{\mathregular{-1}}$"
    nppdata[:]=npp;    
    datafile.close()