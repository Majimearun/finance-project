
# Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from copy import copy
from scipy import stats
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

# Note that the dataframe has to be sorted by date (ascending) and the date column must be named "Date"


#! The date column of the dataframe should be called "Date", or modify functions accordingly

def correct(df):
    """
    Function sorts the DataFrame based on dates and removes all null values
    """
    df = df.sort_values(by = "Date")
    print("DataFrame sorted by Dates in ascending order")
    null = df.isnull().sum()
    if (null.values == 0 ).all() :
        print(f"No null values")
    else :
        df.dropna(inplace = True)
        print(f"Null values removed, remaining number of rows = {len(df)} ")
        
    return df


def basic_info(df):
    """
    A function that takes a stocks comparison dataframe and prints the basic details and returns
    a tuple of (two) dataframes that give statistical and general information about the various columns
    """
    stocks_df = df.copy()
    # not counting the first column as it is Dates
    print(f"number of stocks is : {len(stocks_df.columns[1:])}") 
    # listing out the various companies for which stocks are in our dataframe 
    stocks_df.info()
    # statistical information of the dataframe
    statistics = (stocks_df.describe())
    
    return statistics


def normalize(df):
    """
    Function to normalize the prices based on the initial price
    The function simply divides every stock by it's price at the start date 
    """
    x = df.copy()
    # Loop through each stock (while ignoring time columns with index 0)
    for i in x.columns[1:] :
        x[i] = x[i]/x[i][0]
    return x


def read_dataset(df):
    df = correct(df)
    stats = basic_info(df)
    return df,stats


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