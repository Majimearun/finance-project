import pandas as pd
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.linear_model import Ridge
from stocks_comparison_functions import *
from graphing_functions import *
from data_preprocessing import *

def individual_stock(price_df,vol_df,name):
  return pd.DataFrame({"date": price_df["Date"], "close" : price_df[name], "volume" : vol_df[name]})

def trading_window(data, n = 1):
  data["target"] = data["close"].shift(-n)
  return data

def split_timeseries(X,y,ratio = 0.8):
  split = int(ratio * len(X))
  X_train = X[:split]
  y_train = y[:split]
  X_test = X[split:]
  y_test = y[split:]
  return X_train,y_train,X_test,y_test

def create_stock_df(price_df,vol_df,name,n):
  price_vol_df = individual_stock(price_df,vol_df,name)
  price_vol_target_df = trading_window(price_vol_df,n)
  prediction_df = price_vol_target_df[-n:]
  price_vol_target_df = price_vol_target_df[:-n]

  return prediction_df,price_vol_target_df

def transform_data(df):
  sc_X = MinMaxScaler(feature_range = (0,1))
  sc_y = MinMaxScaler(feature_range = (0,1))
  X = df.iloc[:, 1:-1].values
  y = df.iloc[:, -1].values
  y = y.reshape(len(y),1)
  X = sc_X.fit_transform(X)
  y = sc_y.fit_transform(y) 
  return sc_X,sc_y,X,y

def make_ridge_regrsessor(X_train,y_train,X_test,y_test):
  regressor = Ridge()
  regressor.fit(X_train,y_train)
  lr_accuracy = regressor.score(X_test,y_test)

  return regressor,lr_accuracy

def get_comparison_plot(X,regressor,initial_df):
  predicted_prices = regressor.predict(X)
  predicted = [i[0] for i in predicted_prices ]
  close = [ X[i][0] for i in range(len(X)) ]
  df_predicted = initial_df[["date"]]
  df_predicted.rename(columns = {"date": "Date"}, inplace = True)
  df_predicted["close"] = close
  df_predicted["prediction"] = predicted
  regressor_plot = interactive_plot(df_predicted,"ORIGINAL vs PREDICTION : RIDGE REGRESSION")

  return df_predicted, regressor_plot

def predict_future(prediction_df,sc_y,sc_X,regressor):
  X = prediction_df.iloc[:, 1:-1].values
  results = sc_y.inverse_transform(regressor.predict(sc_X.transform(X)))

  return results

def ridge_regressor(price_df,vol_df,name,ratio,n=1):
  prediction_df,price_vol_target_df = create_stock_df(price_df,vol_df,name,n)
  sc_X,sc_y,X,y = transform_data(price_vol_target_df)
  X_train,y_train,X_test,y_test = split_timeseries(X,y,ratio)
  training_plot = array_plot(X_train,"TRAINING DATA")
  testing_plot = array_plot(X_test,"TESTING DATA")
  regressor,lr_accuracy = make_ridge_regrsessor(X_train,y_train,X_test,y_test)
  df_predicted, regressor_plot = get_comparison_plot(X,regressor,price_vol_target_df)
  results = predict_future(prediction_df,sc_y,sc_X,regressor)
  return regressor,results,regressor_plot,lr_accuracy

