#!/usr/bin/env python

import sys
import json
import os.path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def make_figure(scores, kList, metric, fig_file):
    
    #fig_file = os.path.splitext(jsonFile)[0] + ".png"
    fig_title = os.path.splitext((os.path.basename(fig_file)))[0]
 
    cmap = cm.get_cmap('tab20')
    colors = [cmap(j/20) for j in range(0,20)] 

    f, axs = plt.subplots(2, 6, figsize=(20,10))
    axs = np.concatenate(axs)
    ##axs = list(zip(axs,clf_symbs))
    width = 0.45
    ind = 0

    for algo in scores:

        if "MultinomNB" in algo or "Markov" in algo :
            means = np.array([np.array(scores[algo][str(k)][metric]).mean() for k in kList])
            stds = np.array([np.array(scores[algo][str(k)][metric]).std() for k in kList])

            axs[ind].fill_between(kList, means-stds, means+stds,
                    alpha=0.1, color=colors[ind])
            axs[ind].plot(kList, means, color=colors[ind])

            axs[ind].set_xticks([k for k in kList])
 
            axs[ind].set_title(algo)
            axs[ind].set_ylim([0, 1.1])
            axs[ind].grid()
 
            axs[ind].set_ylabel(metric)
            axs[ind].set_xlabel('Length of k')

            ind +=1
 
    plt.suptitle(fig_title)
    plt.savefig(fig_file)


if __name__ == "__main__":

    scores_file1 = sys.argv[1]
    figure_file = sys.argv[2]
    metric = sys.argv[3]
    str_k_list = sys.argv[4]

    se_ks = str_k_list.split(":")

    if len(se_ks) != 2:
        print("K list argument should contain : to separate start and end")
        sys.exit()

    s_klen = int(se_ks[0])
    e_klen = int(se_ks[1])

    k_main_list = list(range(s_klen, e_klen+1))
 
    scores_1 = json.load(open(scores_file1, "r"))

    make_figure(scores_1, k_main_list, metric, figure_file)

