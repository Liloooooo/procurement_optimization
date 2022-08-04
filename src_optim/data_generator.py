#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 16:35:56 2022

@author: lilowagner
"""


import numpy as np
import pandas as pd 
from datetime import datetime
from scipy.stats import gamma


def get_current_demand():
    """

    get demand array at time self.current_date, which is only on Tuesdays and Thursdays, thus, demand is accumulated up to these days  
    returns: array of shape (2,): demand per group at current time

    """
    
    gamma_shape = np.array((1/8, 1/16, 1/16, 1/16))
    scale = np.array((20, 5, 8, 4))     
    #demands = [round(np.random.gamma(gamma_shape[i], scale[i])) for i in range(4)]
    demands = [round(gamma.rvs(0.6, gamma_shape[i], scale=scale[i])) for i in range(4)]
    return np.array(demands)


def get_demand_periods(periods: int): 
    return [get_current_demand() for i in range(periods)]

def get_data(prices, length, startdate='01-01-2019', save_csv_path = 'toy_data.csv'):
    day_series = pd.Series(pd.date_range(startdate, periods=df_length, freq="d"))
    df=pd.DataFrame(get_demand_periods(df_length), columns = prices, index=day_series)
    df.to_csv(save_csv_path)
    return df
 
    

prices = [18.1, 12.8, 11.2, 14.3]
df_length = 3*365

df=get_data(prices, df_length)




