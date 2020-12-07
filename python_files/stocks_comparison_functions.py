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




def get_returns(df,start_date ,end_date,company,shares):
    """
    Function calculates the dollar return for a particular company during a particular time period
    Returns this value multiplies by the shares owned to get total(net) return for a particular investment

    df : the dataframe containing the stocks data for the companies in question
    start_date : string of the format "YYYY-MM-DD" signifying the start of period of analysis
    end_date : string of the format "YYYY-MM-DD" signifying the end of period of analysis
    company : string/ name of the company where investment was made (should be a column name in the df)
    shares : a float value that is equal to the investment made
    """
    stocks_df = df.copy()
    start =  float(stocks_df.loc[(stocks_df["Date"] == start_date),company])
    end = float(stocks_df.loc[(stocks_df["Date"] == end_date),company]) 
    return (start - end) * shares



def find_max_min_gain_loss(df,start_date,end_date,shares):
    """
    Calls the get_returns function and returns a sorted list of tuples to find the best 
    and worst companies to invest in based on shares invested in each company (during given time period)
    """
    returns = list()
    for i in df.columns[1:] :
        Return = round(get_returns(df,start_date,end_date,i,shares),2)
        returns.append([i,Return])
    sorted_returns = sorted(returns, key=lambda x: x[1])
    return sorted_returns 



def daily_return(df):
    """
    A function that creates and returns a new dataframe that consists of the daily returns of each of the
    companies in the original dataframe
    """
    df_daily_return = df.copy()
    for i in df.columns[1:] :
        for j in range(1,len(df)):
         df_daily_return[i][j] = ((df[i][j]- df[i][j-1])/df[i][j-1]) * 100

        df_daily_return[i][0] = 0
    return df_daily_return



def analyze_daily_returns(df):
    """
    Returns an interactive plot of daily returns vs time , given the stocks comparison dataframe 
    """
    stocks_df = df.copy()
    #* mention that they can double click to isolate a particular trace
    return interactive_plot(daily_return(stocks_df),"Returns vs Time")


def get_correlation(df):
    """
    Returns an the correlation matrix of the various companies, given the stocks comparison dataframe
    """
    stocks_df = df.copy()
    stocks_daily_return = daily_return(stocks_df)
    cm = stocks_daily_return.drop(columns = ["Date"]).corr()
    return cm





def getIndexes(df, value):
    ''' Get index positions of value in dataframe '''
    list_of_index = list()
    # Get bool dataframe with True at positions where the given value exists
    result = df.isin([value])
    # Get list of columns that contains the value
    series = result.any()
    columns = list(series[series == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columns:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            list_of_index.append((row, col))
    # Return a list of tuples indicating the positions of value in the dataframe
    return list_of_index


def n_highest_corr(df,n):
    """
    Returns a list of sets of tuples with the n highest correlations, given the stocks comparison dataframe
    """
    cm = get_correlation(df)
    # removing 1s as that is obviously highest
    ser = cm[cm !=1].max()
    ser = ser.sort_values(ascending = False)
    n_positions = list()
    for i in range(1,n+1) :
        
        list_of_pos = getIndexes(cm,ser[i])
        # removes the reversed duplicates due to (row,col) and (col,row) being the same in a heatmap
        set_of_pos = set(tuple(x) for x in map(sorted, list_of_pos))
        n_positions.append(set_of_pos)
    return n_positions



def n_lowest_corr(df,n):
    """
    Returns a list of sets of tuples with the n lowest correlations, given the stocks comparison dataframe
    """
    cm = get_correlation(df)
    ser = cm.min()
    ser = ser.sort_values(ascending = True)
    n_positions = list()
    for i in range(1,n+1) :
        list_of_pos = getIndexes(cm,ser[i])
        # removes the reversed duplicates due to (row,col) and (col,row) being the same in a heatmap
        set_of_pos = {tuple(item) for item in map(sorted, list_of_pos)}
        n_positions.append(set_of_pos)
    return n_positions

    


