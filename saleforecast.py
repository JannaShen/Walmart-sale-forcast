import os
import pandas as pd
import keras
from sklearn import preprocessing 
import numpy as np
import datetime
# load data into dataframe

path= "D:\work\Big-Mart-Sales-Prediction-master"
filename_train = "\\train\\train.csv"
filename_test = "\\test\\test.csv"
filename_feature = "\\features.csv\\features.csv"
filename_store = "stores.csv"
df_train=pd.read_csv(os.path.realpath(path+filename_train)) # names self-defined names
df_test=pd.read_csv(os.path.realpath(path+filename_test))
df_features=pd.read_csv(os.path.realpath(path+filename_feature))
df_stores=pd.read_csv("stores.csv", header=0)

# explore data

print(df_stores.head(),"\n")
print(df_features.isnull().sum().sort_values(ascending=False),"\n") # find the null value information

#dataset=train.merge(stores, how='left').merge(features, how='left')


def summary_data(data):
    data_type=data.dtypes
    data_missing=data.isnull().sum()
    data_summary=pd.DataFrame({"missing":data_missing,
                               "type":data_type})

    print(data_summary)

# transfor data to datatimes64
df_features['Date']=pd.to_datetime(df_features['Date'],format="%Y-%m-%d")
df_train['Date']=pd.to_datetime(df_train['Date'],format="%Y-%m-%d")
df_test['Date']=pd.to_datetime(df_test['Date'],format="%Y-%m-%d")
     

# determine the training data and testing data
# the given testing data and training data has the store, dept, date
# But we have the features of the store and the date, which are parameters influencing the results and should be taken into the input
# the training data and testing data will combine all thesis features before input into the model

combined_train=pd.merge(df_train,df_stores, how="left", on="Store")
combined_test=pd.merge(df_test,df_stores, how="left", on="Store")
combined_train=pd.merge(combined_train, df_features,how="left", on=["Store","Date"])
combined_test=pd.merge(combined_test, df_features,how="left", on=["Store","Date"])

processed_train = combined_train.fillna(0)
processed_test = combined_test.fillna(0)

processed_train.loc[processed_train['Weekly_Sales'] < 0.0,'Weekly_Sales'] = 0.0
processed_train.loc[processed_train['MarkDown2'] < 0.0,'MarkDown2'] = 0.0
processed_train.loc[processed_train['MarkDown3'] < 0.0,'MarkDown3'] = 0.0

processed_test.loc[processed_test['MarkDown1'] < 0.0,'MarkDown1'] = 0.0
processed_test.loc[processed_test['MarkDown2'] < 0.0,'MarkDown2'] = 0.0
processed_test.loc[processed_test['MarkDown3'] < 0.0,'MarkDown3'] = 0.0
processed_test.loc[processed_test['MarkDown5'] < 0.0,'MarkDown5'] = 0.0

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

processed_train=processed_train[['Store', 'Dept', 'Date', 'Unemployment', 'IsHoliday_x', 'Type', 'Size', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3','MarkDown4', 'MarkDown5', 'CPI', 'Weekly_Sales']]
processed_test=processed_test[['Store', 'Dept', 'Date', 'Unemployment', 'IsHoliday_y', 'Type', 'Size', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3','MarkDown4', 'MarkDown5', 'CPI']]
split_data=datetime.datetime(2012,9,25)
train_set=processed_train.loc[processed_train['Date']<=split_data]
val_set=processed_train.loc[processed_train['Date']>split_data]

train_set=train_set.set_index('Date')
val_set=val_set.set_index('Date')
test_set=processed_test.set_index('Date')

train_set_array = train_set.iloc[:,:].values
val_set_array = val_set.iloc[:,:].values
test_set_array=test_set.iloc[:,:].values

sc=preprocessing.MinMaxScaler(feature_range=(0,1))
train_set_scaled = sc.fit_transform(train_set_array[:,:])
val_set_scaled = sc.fit_transform(val_set_array[:,:])
test_set_scaled=sc.fit_transform(test_set_array[:,:])

x_train=[]
y_train=[]
x_val=[]
y_val=[]
x_test=[]
y_test=[]
x_train, y_train = train_set_scaled[:,:-1], train_set_scaled[:,-1]
x_val, y_val = val_set_scaled[:,:-1], val_set_scaled[:,-1]
x_test=test_set_scaled[:,:]

x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
x_test=x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten

regressor = Sequential()
regressor.add(LSTM(units = 10, return_sequences = True, activation = 'relu', input_shape = (x_train.shape[1], 14)))
regressor.add(Dropout(0.5))
regressor.add(LSTM(units = 10, return_sequences = True, activation = 'relu'))
regressor.add(Dropout(0.5))
regressor.add(LSTM(units = 10, return_sequences = False, activation = 'relu'))
regressor.add(Dropout(0.5))
regressor.add(Dense(units=1, activation = 'sigmoid'))

regressor.compile(optimizer='adam', 
                  loss='mean_squared_error', 
                  metrics=['accuracy'])

history = regressor.fit(x_train, 
              y_train, 
              epochs = 1, 
              batch_size = 512, 
              validation_data = (x_val, y_val),
              verbose = 1)
			 
predicted_sales=regressor.predict(x_test)
x_test = x_test.reshape((x_test.shape[0], x_test.shape[2]))
print(predicted_scales.shape, x_test[:,:].shape)
predicted_weekly_sales=np.concatenate((x_test[:,:],predicted_sales), axis=1)
predicted_weekly_sales=sc.inverse_transform(predicted_weekly_sales)

predicted_weekly_sales=predicted_weekly_sales[:,14:15]

print(predicted_weekly_sales)
sample=processed_test[['Store','Dept','Date']]
sample['ID']=sample['Store'].astype(str)+'_'+sample['Dept'].astype(str)+'_'+sample['Date'].astype(str)
sample['Weekly_Sales']=predicted_weekly_sales
sample.drop(['Store','Dept','Date'])
sample.to_csv("sample.csv",index=False)

