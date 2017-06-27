#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 13:26:48 2017
@author: benharris
"""
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import math
import psycopg2
from scipy.stats import skew,norm
from pandas.tools.plotting import scatter_matrix
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import (LinearRegression, Ridge, 
                                  Lasso, RandomizedLasso,RidgeCV,LassoCV)
from sklearn.feature_selection import RFE, f_regression
from scipy.stats import boxcox
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.metrics.scorer import SCORERS
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV

def constructTopFeatureColumns(features):
    topFeatures = []
    for feature in features:
        if feature[0] > .009:
            topFeatures.append(feature[1])
    return topFeatures

def getTopFeaturesRF(X,y):
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    #df['Gain'] = df['open']-df['close']
    columns = X.columns
    X = scaler_x.fit_transform(X)
    y = scaler_y.fit_transform(y)
    # feature extraction
    model = RandomForestRegressor(n_estimators=300,max_features=80,max_depth=10)
    model.fit(X, y)
    result = sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), columns), reverse=True)
    return constructTopFeatureColumns(result),result

def getTopFeaturesRandomForest(X,y):
    columns = X.columns
    rfe = RandomForestRegressor(n_estimators=300,n_jobs=-1)
    rfe.fit(X,y)
    return rank_to_dict(rfe.feature_importances_, columns)

def getTopFeaturesRidge(X,y):
    columns = X.columns
    ridge = Ridge(alpha=7)
    ridge.fit(X, y)
    return rank_to_dict(np.abs(ridge.coef_), columns)

def getTopFeaturesLasso(X,y):
    columns = X.columns
    lasso = Lasso(alpha=.05)
    lasso.fit(X, y)
    return rank_to_dict(np.abs(lasso.coef_), columns)

def getTopFeaturesRandomLasso(X,y):
    columns = X.columns
    rlasso = RandomizedLasso(alpha=0.04)
    rlasso.fit(X, y)
    return rank_to_dict(np.abs(rlasso.scores_), columns)

def getTopFeaturesLinear(X,y):
    columns = X.columns
    lr = LinearRegression(normalize=True)
    lr.fit(X, y)
    return rank_to_dict(np.abs(lr.coef_), columns)

def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks ))

def getTopFeaturesF(X,y):
    columns = X.columns
    f, pval  = f_regression(X, y, center=True)
    f[np.isnan(f)] = 0
    return rank_to_dict(f, columns)

def getTopFeaturesRFE(X,y):
    columns = X.columns
    lr = LinearRegression()
    rfe = RFE(lr, n_features_to_select=5)
    rfe.fit(X,y)
    return rank_to_dict(list(map(float, rfe.ranking_)), columns, order=-1)

def fixData(df):
    df = df.loc[:, df.isnull().mean() < .1]
    df = df.fillna(df.mean())
    df_fixed = pd.DataFrame(preprocessing.scale(np.sqrt(df)),columns=df.columns)
    df_fixed.index = df.index
    return df_fixed

def getConfidenceInt(df,y):
    mean = df[y].mean()
    std =  df[y].std()
    conf = norm.interval(0.95, loc=mean, scale=std/np.sqrt(132))
    return conf

def getPredictionInt(df,y,z):
    mean = df[y].mean()
    std =  df[y].std()
    mi = mean - z*std
    mx = mean + z*std
    return mi,mx

def shortenColumnNames(columns):
    col = []
    i = 0
    for column in columns:
        col.append(i)
        i+=1
    return col

def getData(sql, conn,index):
    	return pd.read_sql(sql,conn,index_col = index)
    
def dropy(df,y):
    try:
        return df.drop(y, 1)
    except:
        return df

def correlation_matrix(df):
    import seaborn as sns
    # calculate the correlation matrix
    corr = df.corr()
    # plot the heatmap
    sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
    
def analyze(y_test,y_7,name):
  print("LSTM Mean Absolute Error: " +str(round(mean_absolute_error(y_test, y_7),2))+"%")
  print("LSTM R2: " +str(round(r2_score(y_test, y_7),2)))
  print("LSTM RMSE: " +str(mean_squared_error(y_7,y_test)))
  print("")
  fig, ax = plt.subplots()
  fig.suptitle(name)
  ax.scatter(y_test, y_7)
  ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
  ax.set_xlabel('PCI')
  ax.set_ylabel('Predicted PCI')
  plt.show()
  
###################################################
scaler = StandardScaler()
df = pd.read_csv('/Users/benharris/Documents/Projects/sl/train.csv').drop('train_id',axis=1)
df['cat1']=df['subadult_males']+df['adult_males']
df['cat2']=df['adult_females']+df['juveniles']
X = fixData(df[['cat1','cat2','subadult_males','adult_males','adult_females','juveniles']])
y = df['pups']
y = scaler.fit_transform(y)
   
###################################################
####Build Model
###################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_test_scaled = scaler.inverse_transform(y_test)
###################################################
####Lasso
###################################################
lasso = Lasso(alpha=.05)
lasso.fit(X_train, y_train)
score = lasso.score(X_test, y_test)
y_pred = lasso.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
label=("Lasso",score,mean_absolute_error(y_test_scaled,y_pred),mean_squared_error(y_test_scaled,y_pred))
title = "%s R2:%s MAE:%s MSE:%s"%(label[0],round(label[1],2),round(label[2],2),round(label[3],2))
analyze(y_test_scaled,y_pred,title)
###################################################
####RandomForest
###################################################
rfe = RandomForestRegressor(n_estimators=300,n_jobs=-1)
rfe.fit(X_train, y_train)
score = rfe.score(X_test,y_test)
y_pred = rfe.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
label=("Random Forest",score,mean_absolute_error(y_test_scaled,y_pred),mean_squared_error(y_test_scaled,y_pred))
title = "%s R2:%s MAE:%s MSE:%s"%(label[0],round(label[1],2),round(label[2],2),round(label[3],2))
analyze(y_test_scaled,y_pred,title)
###################################################
####Ridge
###################################################
ridge = Ridge(alpha=7)
ridge.fit(X_train, y_train)
score = ridge.score(X_test,y_test)
y_pred = ridge.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
label=("Ridge",score,mean_absolute_error(y_test_scaled,y_pred),mean_squared_error(y_test_scaled,y_pred))
title = "%s R2:%s MAE:%s MSE:%s"%(label[0],round(label[1],2),round(label[2],2),round(label[3],2))
analyze(y_test_scaled,y_pred,title)
#analyze(y_test_scaled,y_pred)
###################################################
####Linear
###################################################
lr = LinearRegression(normalize=True)
lr.fit(X_train, y_train)
score = lr.score(X_test,y_test)
y_pred = lr.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
label=("Linear",score,mean_absolute_error(y_test_scaled,y_pred),mean_squared_error(y_test_scaled,y_pred))
title = "%s R2:%s MAE:%s MSE:%s"%(label[0],round(label[1],2),round(label[2],2),round(label[3],2))
analyze(y_test_scaled,y_pred,title)
###################################################
####Ridge CV
###################################################
scorer = SCORERS['mean_squared_error']
alphas = np.arange(.5,200, 0.5)
rCV = RidgeCV(alphas=alphas, fit_intercept=True, scoring=scorer, normalize=False,cv=10)
rCV.fit(X_train, y_train)
score = rCV.score(X_test,y_test)
y_pred = rCV.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
label=("RidgeCV",score,mean_absolute_error(y_test_scaled,y_pred),mean_squared_error(y_test_scaled,y_pred))
title = "%s R2:%s MAE:%s MSE:%s"%(label[0],round(label[1],2),round(label[2],2),round(label[3],2))
analyze(y_test_scaled,y_pred,title)
###################################################
####Lasso CV
###################################################
scorer = SCORERS['mean_squared_error']
alphas = np.arange(.0001,200, 0.05)
lCV = LassoCV(alphas=alphas,cv=10)
lCV.fit(X_train, y_train)
score = lCV.score(X_test,y_test)
y_pred = lCV.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
label=("LassoCV",score,mean_absolute_error(y_test_scaled,y_pred),mean_squared_error(y_test_scaled,y_pred))
title = "%s R2:%s MAE:%s MSE:%s"%(label[0],round(label[1],2),round(label[2],2),round(label[3],2))
analyze(y_test_scaled,y_pred,title)
###################################################
####Random Forest CV
###################################################
alphas = np.arange(.1,200, 5)
rfe = RandomForestRegressor(n_estimators=200,n_jobs=-1)
param_grid = { 
    'n_estimators': [10,50,100,150,200,250,300,400,700]
}
CV_rfc = GridSearchCV(estimator=rfe, param_grid=param_grid, cv= 10)
CV_rfc.fit(X_train, y_train)
score = CV_rfc.score(X_test,y_test)
y_pred = CV_rfc.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
label=("Random Forest CV",score,mean_absolute_error(y_test_scaled,y_pred),mean_squared_error(y_test_scaled,y_pred))
title = "%s R2:%s MAE:%s MSE:%s"%(label[0],round(label[1],2),round(label[2],2),round(label[3],2))
analyze(y_test_scaled,y_pred,title)