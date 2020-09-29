# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 10:28:46 2020

@author: yimao
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
#import file
crop_df=pd.read_excel(r'C:\Users\yimao\PepsiCo\nyas-challenge-2020-data.xlsx','Crop and Grain Data', usecols='A,B,C,D,E,F,G')
wt_df=pd.read_excel(r'C:\Users\yimao\PepsiCo\nyas-challenge-2020-data.xlsx','Weather Data')
site_df=pd.read_excel(r'C:\Users\yimao\PepsiCo\nyas-challenge-2020-data.xlsx','Site Data')
# Extract Site
crop_df['Site']=crop_df['Site ID'].apply(lambda x: x[:6])
# Extract month_day information to calculate avg feature values
wt_df['month_y']=wt_df['Date (mm/dd/year)'].dt.to_period('M')
wt_df['Site']=wt_df['Site ID'].apply(lambda x: x[:6])
wt_df_avg=wt_df.groupby(['Site ID','month_y']).mean().reset_index()
wt_df_avg['Site']=wt_df_avg['Site ID'].apply(lambda x: x[:6])
# Merge crop and weather feature data
df = pd.merge(crop_df, wt_df,  how='left', left_on=['Site','Assessment Date (mm/dd/year)'], right_on = ['Site','Date (mm/dd/year)'])
# Select those with no weather features
df_null=df[df['Weather Variable A'].isnull()]
df_null['month_y']=df_null['Assessment Date (mm/dd/year)'].dt.to_period('M')
df_null.dropna(axis=1,inplace=True)
# Use month avg weather data to fill in those missing records
df_null_new = pd.merge(df_null, wt_df_avg,  how='left', left_on=['Site','month_y'], right_on = ['Site','month_y'])
df_null_new=df_null_new.dropna()
df_null_new.drop(columns=['month_y'],inplace=True)
df_null_new.drop(columns=['Site ID'],inplace=True)
# Clean up the df dataframe
df.drop(columns=['Date (mm/dd/year)'],inplace=True)
df.drop(columns=['month_y'],inplace=True)
df.drop(columns=['Site ID_y'],inplace=True)
df.dropna(inplace=True)
# append those records using avg weather feature
df_new=pd.concat([df,df_null_new])
df_new['Site ID']=df_new['Site ID_x'].apply(lambda x: x[:7]+x[-4:])
df_new = pd.merge(site_df, df_new,  how='left', on=['Site ID'])
df_new.sort_values(by=['ID'])
# Convert datatype to value
df_new['Variety']=pd.Categorical(df_new['Variety'])
df_new['Variety']=df_new.Variety.cat.codes
df_new['AssessmentType']=pd.Categorical(df_new['Assessment Type'])
df_new['AssessmentType']=df_new.AssessmentType.cat.codes
# clean up values with *
df_new=df_new[df_new['Assessment Score']!='*']
# Use model to train data
X = df_new.iloc[:, np.r_[1:3,5:8,10:12,16:23]]
y = df_new.iloc[:, 14]
X_train, X_test, y_train, y_test = train_test_split(X, y)
#Run selected model (parameter selected by GridSearchCV)
parameters={'n_estimators':[100,200],'max_features':[10,12,14],'max_depth':[3,5,7]}
grid=GridSearchCV(GradientBoostingRegressor(random_state=0),param_grid=parameters)
grid.fit(X_train,y_train)
grid.best_score_, grid.best_estimator_
clf= GradientBoostingRegressor(random_state=0,max_features=12,max_depth=7).fit(X_train, y_train)
clf.score(X_train, y_train),clf.score(X_test, y_test)
#Calculate prediction
df_new['predict']=clf.predict(X)
df_new