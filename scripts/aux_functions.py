''' Script containing helper functions for the final year project.
    
    Please add functions here to use in the notebooks, then add to the import section
    and rerun the notebooks to use them.

    data visualisation functions, and missing percentage checks are included here.

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


''' Data visualisation and cleaning'''
def missing_percentages(X):
    '''
    Check percentage of entries for each variable that contain missing values
    '''
    
    # columns with null values
    X_null = X.columns[X.isnull().any()==True].tolist()
    # null values in these columns
    null_cols = X[X_null].isnull().sum()
    null_percentage = np.round(null_cols/X.shape[0], 2)
    
    df = pd.concat([null_cols, null_percentage], axis=1)
    columns=["total", "percentage"]
    df.columns = columns
    print(df.head(X.shape[0]).sort_values(by='total', ascending = False))


def plot_dist_discrete(var: str, data):
  '''plot distribution of discrete/categorical variable'''
  
  data = data.copy()
  sns.set_style('whitegrid')
  plt.figure(figsize=(8, 6))

  order = sorted(data[var].unique().tolist())
  sns.countplot(data=data, x=var, order=order)
  plt.show()


def plot_dist_continuous(var:str, data: pd.DataFrame, bins: int=50):
  ''' Plot distribution of continuous variable'''
  
  data = data.copy()
  sns.set_style('whitegrid')
  plt.figure(figsize=(8, 6))

  sns.histplot(data=data, x=var, bins=bins)
  plt.show()


def plot_vis_discrete(var: str, data: pd.DataFrame):
  ''' violinplots of visibility distributions aggregated across discrete variable values. '''
  
  data = data.copy()

  sns.set_style('whitegrid')
  plt.figure(figsize=(20, 8))
  
  sns.violinplot(x=var, y='vis', data=data)
  plt.show()


def plot_vis_continuous(var: str, data: pd.DataFrame, alpha: float=0.7):
  ''' Scatter plot of continous variable vs visibility 
      Used to get an idea of the relationship between visibility and 
      continuous variables (e.g., wind speed)
      '''
  
  data = data.copy()
  sns.set_style('whitegrid')
  plt.figure(figsize=(8, 6))

  sns.scatterplot(x=var, y='vis', data=data, alpha=alpha)
  plt.show()


def month_vplot(var: str, data: pd.DataFrame, target='target_hr1', with_target: bool=False):
  ''' violinplot of variable distributions aggregated accross months and separated by
      whether there was fog in the next hour. Best suited to continuous variables,
      not suitable for categorical variables. 
  '''
  
  data = data.copy()
  sns.set_style('whitegrid')
  plt.figure(figsize=(20, 8))

  hue = target if with_target else None
  palette = {"no fog": "orange", "fog": ".85"} if with_target else None
  sns.violinplot(x='month', y=var, data=data, hue=hue, split=with_target,
                   palette=palette)

  plt.title(var+" by month")
  plt.show()

