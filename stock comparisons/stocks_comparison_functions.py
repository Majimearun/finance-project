
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



def statistic_plot(df,fig_title):
    """
    Given a stocks dataframe, this function plots a  simple statistic plot comparing the changing stocks
    of the various columns listed
    A plot is returned
    """
    df.plot(x = "Date", figsize = (15,7), linewidth = "2",title = fig_title)
    plt.grid()
    return plt



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



def interactive_plot(df,title):
    """
    Function to perform an interactive data plotting using plotly express
    Plotly.express module which is imported as px includes functions that can plot interactive plots easily and effectively. 
    Figure instance is returned 
    """
    #* mention that they can double click to isolate a particular trace
    fig = px.line(title = title)

    for i in df.columns[1:] :
        fig.add_scatter(x = df["Date"], y = df[i], name = i)
  
    return fig



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



def correlation_heatmap(df):
    """
    Returns an heatmap of the correlations of the various companies , given the stocks comparison dataframe
    """
    cm = get_correlation(df)
    plt.figure(figsize=(10,10))
    ax = plt.subplot()
    sns.heatmap(cm,annot = True, ax=ax)
    return None


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

    
def static_histogram(df):
    """
    Function plots a static histogram of all listed companies (as column names) in various  subplots, 
    given the stocks comparison dataframe
    """
    stocks_daily_return = daily_return(df)
    stocks_daily_return.hist(figsize = (10,10), bins = 20)
    return None



def interactive_histogram(df):
    """
    Given the stocks comparison dataframe, this function returns an interactive plotly distplot instance
    for easy analysis
    """
    stocks_daily_return = daily_return(df)
    df_hist = stocks_daily_return.copy()
    df_hist = df_hist.drop(columns = ["Date"])
    data = []
    for i in df_hist.columns :
        data.append(stocks_daily_return[i].values)

    #* mention that they can double click to isolate a particular trace
    fig = ff.create_distplot(data,df_hist.columns)

    return fig