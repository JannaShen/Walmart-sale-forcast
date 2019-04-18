#Importing the libraries required
import os
import pandas as pd
import keras
from sklearn import preprocessing 
import numpy as np
import datetime

path= "D:\work\Big-Mart-Sales-Prediction-master"
filename_train = "\\train\\train.csv"
filename_test = "\\test\\test.csv"
filename_feature = "\\features.csv\\features.csv"
filename_store = "stores.csv"


train=pd.read_csv(os.path.realpath(path+filename_train))
test=pd.read_csv(os.path.realpath(path+filename_test))
features=pd.read_csv(os.path.realpath(path+filename_feature))
stores=pd.read_csv("stores.csv", header=0)

# determine the training data and testing data
# the given testing data and training data has the store, dept, date
# But we have the features of the store and the date, which are parameters influencing the results and should be taken into the input
# the training data and testing data will combine all thesis features before input into the model
features.fillna({"MarkDown2":features["MarkDown2"].mean()},inplace=True)
features.fillna({"MarkDown1":features["MarkDown1"].mean()},inplace=True)
features.fillna({"MarkDown3":features["MarkDown3"].mean()},inplace=True)
features.fillna({"MarkDown4":features["MarkDown4"].mean()},inplace=True)
features.fillna({"MarkDown5":features["MarkDown5"].mean()},inplace=True)
features.fillna({"CPI":features["CPI"].mean()},inplace=True)
features.fillna({"Unemployment":features["Unemployment"].mean()},inplace=True)


combined_train=pd.merge(train,stores, how="left", on="Store")
combined_test=pd.merge(test,stores, how="left", on="Store")
processed_train=pd.merge(combined_train, features,how="left", on=["Store","Date"])
processed_test=pd.merge(combined_test, features,how="left", on=["Store","Date"])



# find the null value information

# replace Null values

processed_train.loc[processed_train['Weekly_Sales'] < 0.0,'Weekly_Sales'] = 0.0
processed_train.loc[processed_train['MarkDown2'] < 0.0,'MarkDown2'] = 0.0
processed_train.loc[processed_train['MarkDown3'] < 0.0,'MarkDown3'] = 0.0

processed_test.loc[processed_test['MarkDown1'] < 0.0,'MarkDown1'] = 0.0
processed_test.loc[processed_test['MarkDown2'] < 0.0,'MarkDown2'] = 0.0
processed_test.loc[processed_test['MarkDown3'] < 0.0,'MarkDown3'] = 0.0
processed_test.loc[processed_test['MarkDown5'] < 0.0,'MarkDown5'] = 0.0

print(processed_train.isnull().sum().sort_values(ascending=False),"\n") 
print(processed_test.isnull().sum().sort_values(ascending=False),"\n") 
# explore data
# change datatime
processed_train['Date']=pd.to_datetime(processed_train['Date'],format="%Y-%m-%d")

processed_test['Date']=pd.to_datetime(processed_test['Date'],format="%Y-%m-%d")

print(processed_train.dtypes, processed_test.dtypes)
# transfer types or holiday inyo cartigory
cat_col=["IsHoliday_x", "Type"]
for col in cat_col:
    lbl=preprocessing.LabelEncoder()
    lbl.fit(processed_train[col].values.astype('str'))
    processed_train[col]=lbl.transform(processed_train[col].values.astype('str'))
	
cat_col=["IsHoliday_y", "Type"]
for col in cat_col:
    lbl=preprocessing.LabelEncoder()
    lbl.fit(processed_test[col].values.astype('str'))
    processed_test[col]=lbl.transform(processed_test[col].values.astype('str'))
print(processed_train["IsHoliday_x"], processed_train["Type"], processed_test["IsHoliday_y"], processed_test["Type"])



#Define target and ID columns:
target = 'Weekly_Sales'
IDcol = ['Store','Dept','Date']



from sklearn import metrics
from sklearn.model_selection import cross_val_score
def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target])
    mse=metrics.make_scorer(metrics.mean_squared_error)
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])

    #Perform cross-validation:
    cv_score = cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring=mse)
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print ("\nModel Report")
    print ("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
    print ("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export submission file:
    dtest['ID']=dtest['Store'].astype(str)+'_'+dtest['Dept'].astype(str)+'_'+dtest['Date'].astype(str)
    samplecol=['ID','Weekly_Sales']
    submission = pd.DataFrame({ x: dtest[x] for x in samplecol})
    submission.to_csv(filename, index=False)
    
    
#Liner Regression model
print("Creating the models and processing")
from sklearn.linear_model import LinearRegression, Ridge
predictors = [x for x in processed_train.columns if x not in [target]+IDcol]
# print predictors
alg1 = LinearRegression(normalize=True)
modelfit(alg1, processed_train, processed_test, predictors, target, IDcol, 'sample1.csv')
coef1 = pd.Series(alg1.coef_, predictors).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')

#Ridge Regression Model
predictors = [x for x in processed_train.columns if x not in [target]+IDcol]
alg2 = Ridge(alpha=0.05,normalize=True)
modelfit(alg2, processed_train, processed_test, predictors, target, IDcol, 'sample2.csv')
coef2 = pd.Series(alg2.coef_, predictors).sort_values()
coef2.plot(kind='bar', title='Model Coefficients')
print("Model has been successfully created and trained. The predicted result is in sample2.csv")

# Decision Tree Model

from sklearn.tree import DecisionTreeRegressor
predictors = [x for x in processed_train.columns if x not in [target]+IDcol]
alg3 = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
modelfit(alg3, processed_train, processed_test, predictors, target, IDcol, 'sample3.csv')
coef3 = pd.Series(alg3.feature_importances_, predictors).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances')

print("Model has been successfully created and trained. The predicted result is in sample3.csv")


#Random Forest Model

from sklearn.ensemble import RandomForestRegressor

predictors = [x for x in processed_train.columns if x not in [target]+IDcol]
alg5 = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
modelfit(alg5, processed_train, processed_test, predictors, target, IDcol, 'sample5.csv')
coef5 = pd.Series(alg5.feature_importances_, predictors).sort_values(ascending=False)
coef5.plot(kind='bar', title='Feature Importances')

print("Model has been successfully created and trained. The predicted result is in sample5.csv")

predictors = [x for x in processed_train.columns if x not in [target]+IDcol]
alg6 = RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=100,n_jobs=4)
modelfit(alg6, processed_train, processed_test, predictors, target, IDcol, 'sample6.csv')
coef6 = pd.Series(alg6.feature_importances_, predictors).sort_values(ascending=False)
coef6.plot(kind='bar', title='Feature Importances')

print("Model has been successfully created and trained. The predicted result is in sample6.csv")
