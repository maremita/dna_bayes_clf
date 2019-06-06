#!/usr/bin/env python

import sys
import json
import os.path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def make_figure(scores, kList, jsonFile, verbose=True):
    if verbose: print("\ngenerating a figure", flush=True)
    
    fig_file = os.path.splitext(jsonFile)[0] + ".png"
    fig_title = os.path.splitext((os.path.basename(jsonFile)))[0]
    
    cmap = cm.get_cmap('tab20')
    colors = [cmap(j/20) for j in range(0,20)] 

    f, axs = plt.subplots(5, 4, figsize=(30,20))
    axs = np.concatenate(axs)
    #axs = list(zip(axs,clf_symbs))
    width = 0.45

    for ind, algo in enumerate(scores):
        means = np.array([scores[algo][str(k)][0] for k in kList])
        stds = np.array([scores[algo][str(k)][1] for k in kList])

        axs[ind].fill_between(kList, means-stds, means+stds,
                alpha=0.1, color=colors[ind])
        axs[ind].plot(kList, means, color=colors[ind])

        axs[ind].set_title(algo)
        axs[ind].set_ylim([0, 1.1])
        axs[ind].grid()
        
        if ind%4 == 0:
            axs[ind].set_ylabel('F1 weighted')
        if ind >= 16:
            axs[ind].set_xlabel('K length')
    
        axs[ind].set_xticks([k for k in kList])

    plt.suptitle(fig_title)
    plt.savefig(fig_file)


if __name__ == "__main__":

    scores_file = sys.argv[1]
    str_k_list = sys.argv[2]

    se_ks = str_k_list.split(":")

    if len(se_ks) != 2:
        print("K list argument should contain : to separate start and end")
        sys.exit()

    s_klen = int(se_ks[0])
    e_klen = int(se_ks[1])

    k_main_list = list(range(s_klen, e_klen+1))
 
    the_scores = json.load(open(scores_file, "r"))

    make_figure(the_scores, k_main_list, scores_file)

