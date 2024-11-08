# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 16:04:56 2024
@author: Zhangyinxue

build XGBoost_CbPM model

"""

#coding=utf8
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.inspection import permutation_importance


from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import explained_variance_score, r2_score, mean_absolute_error, mean_squared_error as MSE
import xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score as CVS
import pickle
from scipy.interpolate import interp1d

############################# train model######################################
datapath=r'F:\01NPPpaper\JGROcean\program\basic process\01quasi-measured NPP dataset\modis matched quasi-measured NPP dataset\dataset_profile.xlsx'
data = pd.read_excel(datapath, sheet_name='Sheet2',index_col=0) 
print(data.head())
print(data.shape)
index = data.index
col = data.columns
# split dataset into train and test
data_train, tmp = train_test_split(data, test_size=0.2, random_state=0)
X_train = data_train.iloc[:, 0:14]
y_train = data_train.iloc[:, 14:]
x_tmp=tmp.iloc[:, 0:14]
y_tmp=tmp.iloc[:, 14:]

model = xgboost.XGBRegressor(n_estimators=300,max_depth=5,learning_rate=0.01, random_state=420)    
xgbr = MultiOutputRegressor(model).fit(X_train, y_train)



#################### calculate Permutation Importance##########################
results = permutation_importance(xgbr, X_train, y_train, n_repeats=30, random_state=420)
importances = results.importances_mean
indices = np.argsort(importances)[::-1]
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print(f"{f + 1}. Feature {indices[f]}: {importances[indices[f]]}")



########################## 5-fold crossvalidation##############################
datapath=r'F:\01NPPpaper\JGROcean\审稿过程\一审回复\补充内容\github\dataset_profile.xlsx'
data = pd.read_excel(datapath, sheet_name='XGBoost2',index_col=0) 
print(data.head())
print(data.shape)
index = data.index
col = data.columns
X=data.iloc[:, 0:14]
y=data.iloc[:, 14:]
data_train, data_pre = train_test_split(data, test_size=0.2, random_state=0)
kf=KFold(n_splits=5,shuffle=True)
model = xgboost.XGBRegressor(n_estimators=300,max_depth=5,learning_rate=0.01, random_state=420)       
for train_index, val_index in kf.split(X, y):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]    
    # Train the model on the training set, and then evaluate it on the validation set
    xgbr = MultiOutputRegressor(model).fit(X_train, y_train)
    score = xgbr.score(X_val, y_val) 
    yy=xgbr.predict(X_train)
    RMSE=np.sqrt(MSE(y_train, yy))
    print("Validation score:", score)
    print("RMSE:", RMSE)





########################### plot result of training#############################
x=y_train 
y=xgbr.predict(X_train)
fig1=plt.figure(figsize=(5,5),dpi=300)
ax1=fig1.add_subplot()
plt.scatter(x, y, color='red',s=2)
plt.plot([0, 5000], [0, 5000], color='black', linestyle='-')
plt.xlim([-0.05, 500])
plt.ylim([-0.05, 500])
r_2=r2_score(x, y)
rmse_value = np.sqrt(MSE(x, y))
plt.xlabel('True',fontsize=22,family='Times New Roman')
plt.ylabel('Prediction',fontsize=22,family='Times New Roman')
ax1.text(0.05,0.9,"RMSE="+str(round(rmse_value,2))+" mg C m$^{\mathregular{-3}}$ d$^{\mathregular{-1}}$", transform=ax1.transAxes,fontsize=18,family='Times New Roman')
ax1.text(0.05,0.8,"R\u00b2="+str(round(r_2,2)), transform=ax1.transAxes,fontsize=18,family='Times New Roman')
ax1.text(0.05,0.7,"Num="+str(len(y_train)), transform=ax1.transAxes,fontsize=18,family='Times New Roman')
ax1.text(0.52, 0.1, '(a) Training set', transform=ax1.transAxes,fontsize=20,family='Times New Roman')
plt.yticks(fontsize=16,family='Times New Roman')
plt.xticks(fontsize=16,family='Times New Roman')
plt.show()



############################## plot result of testing###########################
model_pre=xgbr.predict(x_tmp)
fig2=plt.figure(figsize=(5,5),dpi=300)#创建画布
ax1=fig2.add_subplot()
plt.scatter(y_tmp, model_pre, color='red',s=2)
plt.plot([0, 5000], [0, 5000], color='black', linestyle='-')
plt.xlim([-0.05, 500])
plt.ylim([-0.05, 500])
r_2=r2_score(y_tmp, model_pre)
rmse_value = np.sqrt(MSE(y_tmp, model_pre))
plt.xlabel('True',fontsize=22,family='Times New Roman')
plt.ylabel('Prediction',fontsize=22,family='Times New Roman')
ax1.text(0.05,0.9,"RMSE="+str(round(rmse_value,2))+" mg C m$^{\mathregular{-3}}$ d$^{\mathregular{-1}}$", transform=ax1.transAxes,fontsize=18,family='Times New Roman')
ax1.text(0.05,0.8,"R\u00b2="+str(round(r_2,2)), transform=ax1.transAxes,fontsize=18,family='Times New Roman')
ax1.text(0.05,0.7,"Num="+str(len(y_tmp)), transform=ax1.transAxes,fontsize=18,family='Times New Roman')
ax1.text(0.5, 0.1, '(a) Validation set', transform=ax1.transAxes,fontsize=20,family='Times New Roman')
plt.yticks(fontsize=16,family='Times New Roman')
plt.xticks(fontsize=16,family='Times New Roman')
plt.show()


############################## plot result of independent validation##############
datapath2=r'F:\01NPPpaper\JGROcean\program\basic process\02insitu test dataset\result\Mattei2021_test_dataset.xlsx'
data_pred = pd.read_excel(datapath2,sheet_name='PB_ok', index_col=0)
X_validation=data_pred.iloc[:,0:-1]
tep=xgbr.predict(X_validation)  # The results of 32 layers require interpolation and integration
# Define the depth range and interpolation points
depth_range = np.arange(0, 200)  # 0~200m
interp_points = np.arange(0, 200)
x_orig=np.array([0, 1, 2,3,4,5,7,8,10,12,14,17,19,23,27,31,36,41,47,54,61,
                 69,78,87,97,108,120,133,147,163,181,200])
# Initialize an array to store the integration results
integrated_samples = np.zeros(tep.shape[0])
for i in range(tep.shape[0]):
    npp_row = tep[i, :]    
    # Perform linear interpolation to the specified depth
    f = interp1d(x_orig, npp_row, kind='linear')
    interpolated_npp = f(interp_points)   
    # Integration
    integrated_samples[i] = np.trapz(interpolated_npp, x=depth_range)
# save to excel
df = pd.DataFrame({'predict': integrated_samples})
temp = pd.read_excel(datapath2,sheet_name='PB_ok', index_col=0)
temp['predict'] = df['predict'].values
with pd.ExcelWriter(datapath2, engine='openpyxl', mode='a') as writer:
    temp.to_excel(writer, sheet_name='PB_ok2', index=True)
del temp

# plot
temp = pd.read_excel(datapath2,sheet_name='PB_ok2', index_col=0)
y_pred2=temp.iloc[:,-1]
y_validation=temp.iloc[:, -2]
def get_indexes_less_than(numbers, value):
    indexes = []      
    for index, number in enumerate(numbers):
        if number < value:
            indexes.append(index)        
    return indexes  
numbers = y_validation
value = 2900
result = get_indexes_less_than(numbers, value)
y_validation=y_validation.reset_index(level=None, drop=True)
y_pred3=y_pred2.values
y2_validation=y_validation[result]
y2_pred=y_pred3[result]

fig=plt.figure(figsize=(5,5),dpi=300)
ax1=fig.add_subplot()
plt.scatter(y2_validation, y2_pred, color='red',s=35)
plt.plot([0, 3000], [0, 3000], color='black', linestyle='-')
plt.xlim([-0.05, 3000])
plt.ylim([-0.05, 3000])
r_2=r2_score(y2_validation, y2_pred)
rmse_value = np.sqrt(MSE(y2_validation, y2_pred))
plt.xlabel('In situ',fontsize=22,family='Times New Roman') #fontweight="bold"
plt.ylabel('Prediction',fontsize=22,family='Times New Roman')
ax1.annotate("RMSE="+str(round(rmse_value,2))+" mg C m$^{\mathregular{-2}}$ d$^{\mathregular{-1}}$",xy=(100,2700),xytext=(100,2700),fontsize=18,family='Times New Roman')
ax1.annotate("R\u00b2=",xy=(100,2400),xytext=(100,2400),fontsize=18,family='Times New Roman')
ax1.annotate(round(r_2,2),xy=(400,2400),xytext=(400,2400),fontsize=18,family='Times New Roman')
ax1.annotate("Num="+str(len(y2_validation)),xy=(100,2100),xytext=(100,2100),fontsize=18,family='Times New Roman')
ax1.text(0.89, 0.8, '(b)', transform=ax1.transAxes,fontsize=20,family='Times New Roman')
plt.yticks(fontsize=16,family='Times New Roman')
plt.xticks(fontsize=16,family='Times New Roman')
# subplot
ax2=ax1.inset_axes(([0.65, 0.08, 0.3, 0.3]))
ax2.scatter(y_validation, y_pred2, color='red',s=15)
ax2.plot([0, 11000], [0, 11000], color='black', linestyle='-')
ax2.set_xlim([0, 11000])
ax2.set_xlim([0, 11000])
ax2.annotate("Num="+str(len(y_validation)),xy=(300,9000),xytext=(300,9000),fontsize=14,family='Times New Roman')
ax2.tick_params(axis='x', fontsize=12,family='Times New Roman')
ax2.tick_params(axis='y',fontsize=12,family='Times New Roman')
plt.show()



#################  Model hyperparameter tuning  ###############################
# learning curve
def plot_learning_curve(estimator,title, X, y,
    ax=None, #选择子图
    ylim=None, #设置纵坐标的取值范围
    cv=None, #交叉验证
    n_jobs=None #设定索要使用的线程
    ):
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y
    ,shuffle=True
    ,cv=cv
    # ,random_state=420
    ,n_jobs=n_jobs)
    if ax == None:
       ax = plt.gca()
    else:
       ax = plt.figure()
    ax.set_title(title)
    if ylim is not None:
       ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.grid() #绘制网格，不是必须
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-' 
            , color="r",label="Training score")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-'
    , color="g",label="Test score")
    ax.legend(loc="best")
    return ax

cv = KFold(n_splits=5, shuffle = True, random_state=42)
plot_learning_curve(XGBRegressor(n_estimators=400,max_depth=6,learning_rate=0.02, min_child_weight=1,gamma=0,subsample=0.8,\
                    colsample_bytree=0.8,reg_alpha=0,reg_lambda=1)
,"XGB",X_train,y_train,ax=None,cv=cv)
plt.show()



# Observe the effect of parameters on the model using the learning curve
cv = KFold(n_splits=5, shuffle = True, random_state=42)
axisx=range(3,10,1)
rs=[];
for i  in axisx:
    reg= XGBRegressor(n_estimators=750,max_depth=i,learning_rate=0.01, min_child_weight=1,gamma=0,subsample=0.8,\
                        colsample_bytree=0.8,reg_alpha=0,reg_lambda=1,random_state=420)
    rs.append(CVS(reg,X_train,y_train,cv=cv).mean())
print(axisx[rs.index(max(rs))],max(rs))  #返回平均r2,1-偏差就是指r2
plt.figure(figsize=(20,5))
plt.plot(axisx,rs,c="red",label="XGB")
plt.legend()
plt.show()




# Grid search
xgbr = XGBRegressor(n_estimators=550,max_depth=8,learning_rate=0.02, min_child_weight=1,gamma=0,subsample=0.8,\
                    colsample_bytree=0.8,reg_alpha=0,reg_lambda=1)
param_test1 = {
    'n_estimators': np.arange(550,600,10),
    'learning_rate':np.arange(0.01,0.2,0.01)
    }
xgb_res = GridSearchCV(estimator = xgbr, 
                       param_grid = param_test1, 
                       n_jobs=4, 
                       cv=5)
