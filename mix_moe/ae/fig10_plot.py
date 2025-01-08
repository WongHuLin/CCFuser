from typing import Dict
from random import random
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from statistics import mean
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
import matplotlib.ticker as ticker


if __name__ == '__main__':
    
#     df = pd.read_csv("./kernel_eval_nvlink.csv")
#     AG_time = [tmp[2] for tmp in df.loc[df['kernel'] == 'AG'].values.tolist()][0:-1][::-1]
#     GA_time = [tmp[2] for tmp in df.loc[df['kernel'] == 'GA'].values.tolist()][0:-1][::-1]
#     AG_nccl_time = [tmp[2] for tmp in df.loc[df['kernel'] == 'All2All-GEMM'].values.tolist()][0:-1][::-1]
#     GA_nccl_time = [tmp[2] for tmp in df.loc[df['kernel'] == 'GEMM-All2All'].values.tolist()][0:-1][::-1]
#     AG_nccl_comm_time = [tmp[3] for tmp in df.loc[df['kernel'] == 'All2All-GEMM'].values.tolist()][0:-1][::-1]
#     GA_nccl_comm_time = [tmp[3] for tmp in df.loc[df['kernel'] == 'GEMM-All2All'].values.tolist()][0:-1][::-1]
#     AG_nccl_comp_time = [AG_nccl_time[i] - AG_nccl_comm_time[i] for i in range(len(AG_nccl_comm_time))]
#     GA_nccl_comp_time = [GA_nccl_time[i] - GA_nccl_comm_time[i] for i in range(len(GA_nccl_comm_time))]
#     selected_labels = [1- tmp[1] for tmp in df.loc[df['kernel'] == 'GEMM-All2All'].values.tolist()][0:-1][::-1]
    


    
    
#     index = [tmp[3] for tmp in df.loc[df['kernel'] == 'GEMM-All2All'].values.tolist()]
    
#     indeics = [0,11,12,23,24,35,36,47,48,59,60,71]
#     values = range(len(GA_nccl_comm_time))
    
#     o_fig, (plt_result) = \
#             plt.subplots(1, 1, sharex='all', figsize=(8, 4), dpi=600)
    
# #     plt_result.plot(values, AG_time, color='#8ebb76', label="AG kernel", markerfacecolor='#ffbc66')
# #     plt_result.plot(values, GA_time, color='#bf635f', label="GA kernel", markerfacecolor='#00BFFF')
    
#     plt_result.plot(values, AG_time, color='#000000', label="Fused AG", marker='o', markerfacecolor='#00BFFF')
#     plt_result.plot(values, GA_time, color='#000000', label="Fused GA", marker='o', markerfacecolor='#528B8B')
#     plt_result.plot(values, AG_nccl_time, color='#000000', label="Naive AG", marker='o', markerfacecolor='#ffd966')
#     plt_result.plot(values, GA_nccl_time, color='#000000', label="Naive GA", marker='o', markerfacecolor='#bebada')
#     plt_result.set_xticks(values, selected_labels, size=15)

    
#     plt_result.set_xlabel('Data locality rate',fontsize=20)
#     plt_result.set_ylabel('Execution time(ms)',fontsize=20)
    
#     plt.legend(labels=["AG kernel","GA kernel", "Naive AG", "Naive GA"],frameon=False, fontsize=15, ncol=2)
    
#     plt.show()
#     o_fig.savefig('kernel_execution_locality.pdf', dpi=200,bbox_inches = 'tight') 
    
    
    
    
    
    df = pd.read_csv("./output_data/moe_layer_pcie.csv")
    data = []
    data.append([tmp[-3] for tmp in df.loc[df['name'] == 'mix_moe'].values.tolist()])
    data.append([tmp[-2] for tmp in df.loc[df['name'] == 'mix_moe'].values.tolist()])
    data.append([tmp[-1] for tmp in df.loc[df['name'] == 'mix_moe'].values.tolist()])
    
    data.append([tmp[-3] for tmp in df.loc[df['name'] == 'faster_moe'].values.tolist()])
    data.append([tmp[-2] for tmp in df.loc[df['name'] == 'faster_moe'].values.tolist()])
    data.append([tmp[-1] for tmp in df.loc[df['name'] == 'faster_moe'].values.tolist()])
    
    data.append([tmp[-3] for tmp in df.loc[df['name'] == 'fast_moe'].values.tolist()])
    data.append([tmp[-2] for tmp in df.loc[df['name'] == 'fast_moe'].values.tolist()])
    data.append([tmp[-1] for tmp in df.loc[df['name'] == 'fast_moe'].values.tolist()])
    
    for i in range(len(data[0])):
        for j in range(len(data)//3):
            data[j*3 + 0][i] = data[6][i] / data[j*3 + 0][i]
            data[j*3 + 1][i] = data[7][i] / data[j*3 + 1][i]
            data[j*3 + 2][i] = data[8][i] / data[j*3 + 2][i]

    save_file_name = ["moe_layer_total_pcie.pdf", "moe_layer_forward_pcie.pdf", "moe_layer_backward_pcie.pdf"]
    print(data)
    print(mean(data[0]), mean(data[1]),mean(data[2]),)
    
    
    
    tmp = [data[0][i]/data[3][i] for i in range(len(data[0]))]
    print(tmp)
    print(mean(tmp))
    for i in range(len(data)//3):
        o_fig, (plt_result) = \
            plt.subplots(1, 1, sharex='all', figsize=(8, 4), dpi=600)
    
        selected_x_labels = [tmp[1] for tmp in df.loc[df['name'] == 'mix_moe'].values.tolist()]
        values = range(len(data[0]))
        bar_width = 0.2
        bar_space = 0.0
        values_minus_barwidth=[i - bar_width - bar_space for i in values]
        values_plus_barwidth=[i + bar_width + bar_space for i in values]
    
        bar1 = plt.bar(values_minus_barwidth, data[6+i], bar_width, alpha=alphas[1], hatch=hatchs[-2], label="FastMoe",align="center", color=colors[1], edgecolor="black")
        bar2 = plt.bar(values, data[3+i], bar_width, alpha=alphas[3], hatch=hatchs[1], label="FasterMoe",align="center", color=colors[3],edgecolor="black")
        bar3 = plt.bar(values_plus_barwidth, data[i], bar_width, alpha=alphas[3], hatch=hatchs[2], label="CCFuser",align="center", color=colors[4],edgecolor="black")

        plt.legend(labels=["FastMoe","FasterMoe", "CCFuser"],frameon=False, fontsize=15,  ncol=3)
        plt_result.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

        plt_result.set_xticks(values, selected_x_labels, size=15)
        plt_result.set_xlabel('Number of experts per GPU',fontsize=20)
        plt_result.set_ylabel('Speedup',fontsize=20)
        plt.show()
        o_fig.savefig(save_file_name[i], dpi=200,bbox_inches = 'tight') 
        
    
    
    # df = pd.read_csv("./moe_layer_nvlink.csv")
    # data = []
    # data.append([tmp[-3] for tmp in df.loc[df['name'] == 'mix_moe'].values.tolist()])
    # data.append([tmp[-2] for tmp in df.loc[df['name'] == 'mix_moe'].values.tolist()])
    # data.append([tmp[-1] for tmp in df.loc[df['name'] == 'mix_moe'].values.tolist()])
    
    # data.append([tmp[-3] for tmp in df.loc[df['name'] == 'faster_moe'].values.tolist()])
    # data.append([tmp[-2] for tmp in df.loc[df['name'] == 'faster_moe'].values.tolist()])
    # data.append([tmp[-1] for tmp in df.loc[df['name'] == 'faster_moe'].values.tolist()])
    
    # data.append([tmp[-3] for tmp in df.loc[df['name'] == 'fast_moe'].values.tolist()])
    # data.append([tmp[-2] for tmp in df.loc[df['name'] == 'fast_moe'].values.tolist()])
    # data.append([tmp[-1] for tmp in df.loc[df['name'] == 'fast_moe'].values.tolist()])
    
    # for i in range(len(data[0])):
    #     for j in range(len(data)//3):
    #         data[j*3 + 0][i] = data[6][i] / data[j*3 + 0][i]
    #         data[j*3 + 1][i] = data[7][i] / data[j*3 + 1][i]
    #         data[j*3 + 2][i] = data[8][i] / data[j*3 + 2][i]

    # save_file_name = ["moe_layer_total_nvlink.pdf", "moe_layer_forward_nvlink.pdf", "moe_layer_backward_nvlink.pdf"]
    # print(data)
    # from statistics import mean
    
    # print(mean(data[0]), mean(data[1]),mean(data[2]),)
    # tmp = [data[0][i]/data[3][i] for i in range(len(data[0]))]
    # print(tmp)
    
    # print(mean(tmp))
    # for i in range(len(data)//3):
    #     o_fig, (plt_result) = \
    #         plt.subplots(1, 1, sharex='all', figsize=(8, 4), dpi=600)
    
    #     selected_x_labels = [tmp[1] for tmp in df.loc[df['name'] == 'mix_moe'].values.tolist()]
    #     values = range(len(data[0]))
    #     bar_width = 0.2
    #     bar_space = 0.0
    #     values_minus_barwidth=[i - bar_width - bar_space for i in values]
    #     values_plus_barwidth=[i + bar_width + bar_space for i in values]
    
    #     bar1 = plt.bar(values_minus_barwidth, data[6+i], bar_width, alpha=alphas[1], hatch=hatchs[-2], label="FastMoe",align="center", color=colors[1], edgecolor="black")
    #     bar2 = plt.bar(values, data[3+i], bar_width, alpha=alphas[3], hatch=hatchs[1], label="FasterMoe",align="center", color=colors[3],edgecolor="black")
    #     bar3 = plt.bar(values_plus_barwidth, data[i], bar_width, alpha=alphas[3], hatch=hatchs[2], label="CCFuser",align="center", color=colors[4],edgecolor="black")

    #     plt.legend(labels=["FastMoe","FasterMoe", "CCFuser"],frameon=False, fontsize=15,  ncol=3)
    #     plt_result.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    #     plt_result.set_xticks(values, selected_x_labels, size=15)
    #     plt_result.set_xlabel('Number of experts per GPU',fontsize=20)
    #     plt_result.set_ylabel('Speedup',fontsize=20)
    #     plt.show()
    #     o_fig.savefig(save_file_name[i], dpi=200,bbox_inches = 'tight') 
    
#     GA_time = [tmp[2] for tmp in df.loc[df['kernel'] == 'GA'].values.tolist()]
#     AG_nccl_time = [tmp[2] for tmp in df.loc[df['kernel'] == 'All2All-GEMM'].values.tolist()]
#     GA_nccl_time = [tmp[2] for tmp in df.loc[df['kernel'] == 'GEMM-All2All'].values.tolist()]
#     AG_nccl_comm_time = [tmp[3] for tmp in df.loc[df['kernel'] == 'All2All-GEMM'].values.tolist()]
#     GA_nccl_comm_time = [tmp[3] for tmp in df.loc[df['kernel'] == 'GEMM-All2All'].values.tolist()]
#     AG_nccl_comp_time = [AG_nccl_time[i] - AG_nccl_comm_time[i] for i in range(len(AG_nccl_comm_time))]
#     GA_nccl_comp_time = [GA_nccl_time[i] - GA_nccl_comm_time[i] for i in range(len(GA_nccl_comm_time))]
#     selected_labels = [int(tmp[-1]) for tmp in df.loc[df['kernel'] == 'GEMM-All2All'].values.tolist()]
    
#     o_fig, (plt_result) = \
#             plt.subplots(1, 1, sharex='all', figsize=(8, 4), dpi=600)
    
    
#     values = range(len(selected_labels))
#     bar_width = 0.2
#     bar_space = 0.0
#     values_minus_barwidth=[i - bar_width/2 - bar_space for i in values]
#     values_minus2_barwidth=[i - bar_width/2 - bar_width - bar_space for i in values]
#     values_plus_barwidth=[i + bar_width/2 + bar_space for i in values]
#     values_plus2_barwidth=[i + bar_width/2 + bar_width + bar_space for i in values]
    
#     print(AG_time, GA_time)
    
#     bar1 = plt.bar(values_minus2_barwidth, AG_time, bar_width, alpha=alphas[1], hatch=hatchs[-2], label="Fused AG",align="center", color=colors[1], edgecolor="black")
#     bar3 = plt.bar(values_minus_barwidth, AG_nccl_comp_time, bar_width, alpha=alphas[3], hatch=hatchs[1], label="Naive AG",align="center", color=colors[3],edgecolor="black")
#     bar2 = plt.bar(values_plus_barwidth, GA_time, bar_width, alpha=alphas[3], hatch=hatchs[2], label="Fused GA",align="center", color=colors[4],edgecolor="black")
#     bar5 = plt.bar(values_plus2_barwidth, GA_nccl_comp_time, bar_width, alpha=alphas[3], hatch=hatchs[0], label="Naive GA",align="center", color=colors[5],edgecolor="black")

#     plt.legend(labels=["Fused AG","Fused GA", "Naive AG-comm", "Naive AG", "Naive GA",  "Naive GA-comm"],frameon=False, fontsize=15,  bbox_to_anchor=(0.06, 1, 0.95, 0), ncol=2)
    
#     plt_result.set_xticks(values, selected_labels, size=15)
#     plt_result.set_xlabel('Number of experts per GPU',fontsize=20)
#     plt_result.set_ylabel('Execution time(ms)',fontsize=20)
#     o_fig.savefig('kernel_execution_expert.pdf', dpi=200,bbox_inches = 'tight') 


    
#     bar6 = plt.bar(values_plus_barwidth, GA_nccl_comm_time, bar_width,bottom=GA_nccl_comp_time, alpha=alphas[0], hatch=hatchs[2], label="Other operations",align="center",  color= colors[2], edgecolor="black")
    
#     sync_data = df.loc[df['model_name'] == 'raptor_t']
#     sync_data = sync_data.loc[sync_data['async'] == False].values.tolist()
#     selected_labels = ["("+str(sync_data[i][1])+","+str(sync_data[i][3])+")" for i in range(len(sync_data))]
#     sparse_data_sync_buttom = [(sync_data[i][5]-sync_data[i][6])/sync_data[i][5] for i in range(len(sync_data))]
#     sparse_data_sync_top = [sync_data[i][6]/sync_data[i][5] for i in range(len(sync_data))]
    
#     async_data = df.loc[df['model_name'] == 'raptor_t']
#     async_data = async_data.loc[async_data['async'] == True].values.tolist()
#     sparse_data_asyc_buttom = [(async_data[i][5]-async_data[i][6])/async_data[i][5] for i in range(len(async_data))]
#     sparse_data_asyc_top = [async_data[i][6]/async_data[i][5] for i in range(len(async_data))]
    
#     bigbird_data = df.loc[df['model_name'] == 'pytorch'].values.tolist()
#     bigbird_buttom = [(bigbird_data[i][5]-bigbird_data[i][6])/bigbird_data[i][5] for i in range(len(bigbird_data))]
#     bigbird_top = [bigbird_data[i][6]/bigbird_data[i][5] for i in range(len(bigbird_data))]
    
#     sync_total_excution_time = [sync_data[i][5] for i in range(len(sync_data))]
#     async_total_excution_time = [async_data[i][5] for i in range(len(async_data))]
#     bigbird_total_excution_time = [bigbird_data[i][5] for i in range(len(bigbird_data))]
#     print(sync_total_excution_time)
#     print(async_total_excution_time)
#     print(bigbird_total_excution_time)

    
#     bar1 = plt.bar(values_minus_barwidth, sparse_data_asyc_buttom, bar_width, alpha=alphas[1], hatch=hatchs[-1], label="Async indexing",align="center", color=colors[1], edgecolor="black")
#     bar5 = plt.bar(values_plus_barwidth, bigbird_buttom, bar_width, alpha=alphas[3], hatch=hatchs[-1], label="Naive implementation",align="center", color=colors[3],edgecolor="black")
#     bar3 = plt.bar(values, sparse_data_sync_buttom, bar_width, alpha=alphas[5], hatch=hatchs[-1], label="Sync indexing",align="center", color=colors[4], edgecolor="black")
#     bar2 = plt.bar(values_minus_barwidth, sparse_data_asyc_top,bar_width, sparse_data_asyc_buttom, alpha=alphas[0], hatch=hatchs[-1], label="GPU idle proportion",align="center", color= 'none', edgecolor="black",linestyle='dashed')
#     bar4 = plt.bar(values, sparse_data_sync_top, bar_width,bottom=sparse_data_sync_buttom, alpha=alphas[0], hatch=hatchs[-1], label="Other operations",align="center",  color= 'none', edgecolor="black",linestyle='dashed')
#     bar6 = plt.bar(values_plus_barwidth, bigbird_top, bar_width,bottom=bigbird_buttom, alpha=alphas[0], hatch=hatchs[-1], label="Other operations",align="center",  color= 'none', edgecolor="black",linestyle='dashed')
#     plt.legend(labels=["Async indexing","Naive implementation" ,"Sync indexing"],frameon=False, fontsize=15, columnspacing=3, bbox_to_anchor=(0.06, 1, 0.95, 0), ncol=2)
   
#     plt.ylim((0, 1.35))
    
#     plt_result.set_ylabel('GPU Utilization', size=15)
#     plt_result.set_xticks(values_ticks, selected_labels, size=15)
#     plt.xticks(rotation=45)
    
#     ax2 = plt.twinx()
#     ax2.set_ylabel('Total execution time(ms)', size=15)
#     plt.ylim((0, 1300))
#     plt.plot(values, async_total_excution_time, color='#000000', label="Async processing", marker='o', markerfacecolor='#ffbc66')
#     plt.plot(values, bigbird_total_excution_time, color='#000000', label="Naive processing", marker='o', markerfacecolor='#00BFFF')
#     plt.plot(values, sync_total_excution_time, color='#000000', label="Sync processing", marker='o', markerfacecolor='#528B8B')
# #     plt.legend(labels=["","",""],frameon=False, fontsize=15, columnspacing=0.7, bbox_to_anchor=(0.06, 1, 0.95, 0), ncol=2)
#     plt.legend(labels=["","",""],frameon=False, fontsize=15, columnspacing=12, bbox_to_anchor=(-0.22, 1, 0.95, 0), ncol=2)
    
#     o_fig.savefig('asyc_generation.pdf', dpi=200,bbox_inches = 'tight')  # 保存图片
    
    

#     data = {}
#     for i in range(4):
#         tmp = {}
#         tmp['local_data'] = [tmp[1] for tmp in df.loc[df['gpu_id'] == i].values.tolist()]
#         tmp['max_data'] = [tmp[2] for tmp in df.loc[df['gpu_id'] == i].values.tolist()]
#         data[i] = tmp

    
#     o_fig, (plt_result) = \
#             plt.subplots(1, 1,  figsize=(6, 4), dpi=600, sharex=True, sharey=True)
#     values = range(len(data[0]['local_data']))
    
#     plt_result.set_yticks([0, 0.5, 1])
    
    
#     # 设置x和y轴标签
#     plt_result.set_ylabel('Data localization rate',fontsize=20)
#     plt_result.set_xlabel('Iteration',fontsize=20)


#     plt_result.plot(values, data[3]['local_data'], color='#8ebb76', label="local_data", markerfacecolor='#ffbc66')
#     plt_result.plot(values, data[3]['max_data'], color='#bf635f', label="max_data", markerfacecolor='#00BFFF')

#     plt.legend(labels=["Rl", "Rt"], fontsize=20)



    

# # #     plt.legend(labels=["","",""],frameon=False, fontsize=15, columnspacing=12, bbox_to_anchor=(-0.22, 1, 0.95, 0), ncol=2)
    
# #     o_fig.savefig('motivation.pdf', dpi=200,bbox_inches = 'tight')  # 保存图片
    
# #     plt.show()
    