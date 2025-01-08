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
hatchs = [    '//',     '\\\\',      '|||',     '--',     'xx',     '++',     '..',     '+++',    '']



import csv

if __name__ == '__main__':
    
    df = pd.read_csv("./output_data/kernel_eval_nvlink.csv")
    AG_time = [tmp[2] for tmp in df.loc[df['kernel'] == 'AG'].values.tolist()][0:-1][::-1]
    GA_time = [tmp[2] for tmp in df.loc[df['kernel'] == 'GA'].values.tolist()][0:-1][::-1]
    AG_nccl_time = [tmp[2] for tmp in df.loc[df['kernel'] == 'All2All-GEMM'].values.tolist()][0:-1][::-1]
    GA_nccl_time = [tmp[2] for tmp in df.loc[df['kernel'] == 'GEMM-All2All'].values.tolist()][0:-1][::-1]
    AG_nccl_comm_time = [tmp[3] for tmp in df.loc[df['kernel'] == 'All2All-GEMM'].values.tolist()][0:-1][::-1]
    GA_nccl_comm_time = [tmp[3] for tmp in df.loc[df['kernel'] == 'GEMM-All2All'].values.tolist()][0:-1][::-1]
    AG_nccl_comp_time = [AG_nccl_time[i] - AG_nccl_comm_time[i] for i in range(len(AG_nccl_comm_time))]
    GA_nccl_comp_time = [GA_nccl_time[i] - GA_nccl_comm_time[i] for i in range(len(GA_nccl_comm_time))]
    selected_labels = [1- tmp[1] for tmp in df.loc[df['kernel'] == 'GEMM-All2All'].values.tolist()][0:-1][::-1]
    


    
    
    index = [tmp[3] for tmp in df.loc[df['kernel'] == 'GEMM-All2All'].values.tolist()]
    
    values = range(len(GA_nccl_comm_time))
    
    o_fig, (plt_result) = \
            plt.subplots(1, 1, sharex='all', figsize=(8, 4), dpi=600)
    
#     plt_result.plot(values, AG_time, color='#8ebb76', label="AG kernel", markerfacecolor='#ffbc66')
#     plt_result.plot(values, GA_time, color='#bf635f', label="GA kernel", markerfacecolor='#00BFFF')
    
    plt_result.plot(values, AG_time, color='#000000', label="Fused AG", marker='o', markerfacecolor='#00BFFF')
    plt_result.plot(values, GA_time, color='#000000', label="Fused GA", marker='o', markerfacecolor='#528B8B')
    plt_result.plot(values, AG_nccl_time, color='#000000', label="Naive AG", marker='o', markerfacecolor='#ffd966')
    plt_result.plot(values, GA_nccl_time, color='#000000', label="Naive GA", marker='o', markerfacecolor='#bebada')
    plt_result.set_xticks(values, selected_labels, size=15)

    
    plt_result.set_xlabel('Data locality rate',fontsize=20)
    plt_result.set_ylabel('Execution time(ms)',fontsize=20)
    plt_result.set_yticks([2, 4, 6])
    plt.legend(labels=["AG kernel","GA kernel", "Naive AG", "Naive GA"],frameon=False, fontsize=15, ncol=2)
    plt.ylim(0, 7)
    plt.show()
    o_fig.savefig('kernel_execution_locality.pdf', dpi=200,bbox_inches = 'tight') 
    
    
    
    
    
    df = pd.read_csv("./output_data/kernel_eval_expert.csv")
    AG_time = [tmp[2] for tmp in df.loc[df['kernel'] == 'AG'].values.tolist()]
    GA_time = [tmp[2] for tmp in df.loc[df['kernel'] == 'GA'].values.tolist()]
    AG_nccl_time = [tmp[2] for tmp in df.loc[df['kernel'] == 'All2All-GEMM'].values.tolist()]
    GA_nccl_time = [tmp[2] for tmp in df.loc[df['kernel'] == 'GEMM-All2All'].values.tolist()]
    AG_nccl_comm_time = [tmp[3] for tmp in df.loc[df['kernel'] == 'All2All-GEMM'].values.tolist()]
    GA_nccl_comm_time = [tmp[3] for tmp in df.loc[df['kernel'] == 'GEMM-All2All'].values.tolist()]
    AG_nccl_comp_time = [AG_nccl_time[i] - AG_nccl_comm_time[i] for i in range(len(AG_nccl_comm_time))]
    GA_nccl_comp_time = [GA_nccl_time[i] - GA_nccl_comm_time[i] for i in range(len(GA_nccl_comm_time))]
    selected_labels = [int(tmp[-1]) for tmp in df.loc[df['kernel'] == 'GEMM-All2All'].values.tolist()]
    
    o_fig, (plt_result) = \
            plt.subplots(1, 1, sharex='all', figsize=(8, 4), dpi=600)
    
    
    values = range(len(selected_labels))
    bar_width = 0.2
    bar_space = 0.0
    values_minus_barwidth=[i - bar_width/2 - bar_space for i in values]
    values_minus2_barwidth=[i - bar_width/2 - bar_width - bar_space for i in values]
    values_plus_barwidth=[i + bar_width/2 + bar_space for i in values]
    values_plus2_barwidth=[i + bar_width/2 + bar_width + bar_space for i in values]
    
    print(AG_time, GA_time)
    
    bar1 = plt.bar(values_minus2_barwidth, AG_time, bar_width, alpha=alphas[1], hatch=hatchs[-2], label="Fused AG",align="center", color=colors[1], edgecolor="black")
    bar3 = plt.bar(values_minus_barwidth, AG_nccl_comp_time, bar_width, alpha=alphas[3], hatch=hatchs[1], label="Naive AG",align="center", color=colors[3],edgecolor="black")
    bar4 = plt.bar(values_minus_barwidth, AG_nccl_comm_time, bar_width,bottom=AG_nccl_comp_time, alpha=alphas[0], hatch=hatchs[1], label="Naive AG-comm",align="center",  color= 'none', edgecolor="black",linestyle='dashed')
    bar2 = plt.bar(values_plus_barwidth, GA_time, bar_width, alpha=alphas[3], hatch=hatchs[2], label="Fused GA",align="center", color=colors[4],edgecolor="black")
    bar5 = plt.bar(values_plus2_barwidth, GA_nccl_comp_time, bar_width, alpha=alphas[3], hatch=hatchs[0], label="Naive GA",align="center", color=colors[5],edgecolor="black")
    bar6 = plt.bar(values_plus2_barwidth, GA_nccl_comm_time, bar_width,bottom=GA_nccl_comp_time, alpha=alphas[0], hatch=hatchs[0], label="Naive GA-comm",align="center",  color= 'none', edgecolor="black",linestyle='dashed')

    plt.legend(labels=["Fused AG","Fused GA", "Naive AG-comm", "Naive AG", "Naive GA",  "Naive GA-comm"],frameon=False, fontsize=15,  bbox_to_anchor=(0.06, 1, 0.95, 0), ncol=2)
    
    plt_result.set_xticks(values, selected_labels, size=15)
    plt_result.set_xlabel('Number of experts per GPU',fontsize=20)
    plt_result.set_ylabel('Execution time(ms)',fontsize=20)
    o_fig.savefig('kernel_execution_expert.pdf', dpi=200,bbox_inches = 'tight') 


    