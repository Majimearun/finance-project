import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from copy import copy
from scipy import stats
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go


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



def correlation_heatmap(df):
    """
    Returns an heatmap of the correlations of the various companies , given the stocks comparison dataframe
    """
    cm = get_correlation(df)
    plt.figure(figsize=(10,10))
    ax = plt.subplot()
    sns.heatmap(cm,annot = True, ax=ax)
    return None


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



def statistic_plot(df,fig_title):
    """
    Given a stocks dataframe, this function plots a  simple statistic plot comparing the changing stocks
    of the various columns listed
    A plot is returned
    """
    df.plot(x = "Date", figsize = (15,7), linewidth = "2",title = fig_title)
    plt.grid()
    return plt

def array_plot(data,title):
  plt.figure(figsize = (13,5))
  plt.plot(data,linewidth = 3)
  plt.title(title)
  plt.grid()
  return plt