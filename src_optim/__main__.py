#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 12:29:48 2022

@author: lilowagner
"""

from greedy_agent import GreedyAgent
from environment import ProcurementEnvironment
import pandas as pd
import matplotlib.pyplot as plt 

env_info={'start_date': '010119', 
          'supplier_index': 15.4, 
          'VLT': 3, 
          'initial_stock': 22}

df = pd.read_csv('toy_data.csv', index_col=0)
env = ProcurementEnvironment(dataset = df, env_info=env_info)
greedy_agent = GreedyAgent(environment=env)

epochs = 2
results = greedy_agent.test(epochs=epochs)

fig, axs =plt.subplots(nrows=epochs, sharex=True)
for i in range(epochs): 
    y = [val[i] for val in results['constraints'].values()]
    labels = [key for key in results['constraints'].keys()]
    axs[i].bar(range(len(df.columns)), y, tick_label=labels)
    plt.xticks(rotation = 20)
    axs[i].set_title('How constraints apply in epoch {}'.format(i+1))