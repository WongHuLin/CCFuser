from typing import Dict
from random import random
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator

# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
# plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))


color_def=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#bdd7ee', '#8dd3c7', '#bebada', '#fb8072', '#80b1d3']
config = {
        "font.family":'serif',
        "font.size": 18,
        "mathtext.fontset":'stix',
        "font.serif": ['Times New Roman']#['SimSun'],
          }    #mac字体不一样   Songti SC  windows Simsun
plt.rcParams['lines.linewidth'] = 2
# 显示网格线
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['hatch.color'] = 'black'
rcParams.update(config)
rcParams.update(config)

colors = [    '#f4b183',     '#ffd966',    '#c5e0b4',    '#bdd7ee',    "#8dd3c7",    "#bebada",    "#fb8072",    "#80b1d3"]
alphas = [1, 1, 1, 1,1,1,1,1,1]
hatchs = [    '//',     '\\',      '||',     '--',     'xx',     '+',     '..',     '+++',    '']

import csv

if __name__ == '__main__':
    
    df = pd.read_csv("./output_data/e2e_model.csv")
    data = []
    data.append(df.loc[df['name'] == 'fastermoe'].values.tolist()[0][1:])
    data.append(df.loc[df['name'] == 'mix_moe'].values.tolist()[0][1:])
    data.append(df.loc[df['name'] == 'naive'].values.tolist()[0][1:])
    selected_x_labels = ['MOE-GPT', 'MOE-BERT', 'MOE-Transformer-xl']
    
    o_fig, (plt_result) = \
            plt.subplots(1, 1, sharex='all', figsize=(7, 3), dpi=600)
    
    tmp = [round(data[0][i]/data[1][i],2) for i in range(len(data[0]))]
    print(tmp)
    
    values = range(len(data[0]))
    bar_width = 0.2
    bar_space = 0.0
    values_minus_barwidth=[i - bar_width - bar_space for i in values]
    values_plus_barwidth=[i + bar_width + bar_space for i in values]
    labels = ["FasterMOE","CCFuser", "Naive"]
    
    bar1 = plt.bar(values_minus_barwidth, data[0], bar_width, alpha=alphas[1], hatch=hatchs[-4], label="FasterMOE",align="center", color=colors[1], edgecolor="black")
    bar2 = plt.bar(values, data[1], bar_width, alpha=alphas[3], hatch=hatchs[1], label="CCFuser",align="center", color=colors[3],edgecolor="black")
    bar3 = plt.bar(values_plus_barwidth, data[2], bar_width, alpha=alphas[3], hatch=hatchs[0], label="Naive",align="center", color=colors[4],edgecolor="black")
    plt.legend(labels=["FasterMOE","CCFuser", "Naive"],frameon=False, fontsize=15,  ncol=3)

#     plt_result.set_xlabel('Model name',fontsize=20)

    
    # 在柱状图上标注带箭头和虚线的差距
    for i in range(len(labels)):
        y1 = data[0][i]
        y2 = data[1][i]
        diff = y1 - y2
    
        # 绘制虚线
        plt_result.hlines(y1, values[i] - bar_width/2, values[i] + bar_width/2, colors='gray', linestyles='dashed')
        plt_result.hlines(y2, values[i] - bar_width/2, values[i] + bar_width/2, colors='gray', linestyles='dashed')
    
        # 标注差距的箭头
        plt_result.annotate('',
                xy=(values[i], y2),           # 箭头的终点
                xytext=(values[i], y1 ),        # 箭头的起点
                arrowprops=dict(arrowstyle='<->', color='gray', linestyle='dashed'),
                ha='center', va='bottom')
        plt_result.text(values[i]+bar_width, (y1 + y2) / 2, f'{tmp[i]}x', color='black', ha='center', va='center')
    
    plt_result.set_yticks([0, 200, 400, 600])
    plt_result.set_xticks(values, selected_x_labels, size=15)
    plt_result.set_ylabel('Step time (ms)',fontsize=20)
    plt.show()
    o_fig.savefig('e2e_model.pdf', dpi=200,bbox_inches = 'tight') 
        
