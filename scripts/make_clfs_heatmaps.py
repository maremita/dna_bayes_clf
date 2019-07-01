#!/usr/bin/env python

import sys
import json
import os.path
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns


def compile_data(json_sc, clf_kwds, kList, metric):
    scores_tmp = defaultdict(dict)
    scores = dict()

    for algo in json_sc:
        for kwd in clf_kwds: 
            if kwd in algo:
                scores_tmp[kwd][algo] = np.array([np.array(json_sc[algo][str(k)][metric]).mean() for k in kList])

    for kwd in clf_kwds:
        scores[kwd] = pd.DataFrame.from_dict(scores_tmp[kwd], orient='index')
        scores[kwd].columns = kList
 
    return scores


def make_figure(scores1, scores2, kList, metric, fig_file):
    
    fig_title = os.path.splitext((os.path.basename(fig_file)))[0]
 
    cmap = cm.get_cmap('tab20')
    colors = [cmap(j/20) for j in range(0,20)] 

    f, axs = plt.subplots(2, 2, figsize=(20,6))
    axs = np.concatenate(axs)
    axs[0].get_shared_y_axes().join(axs[1])
    axs[2].get_shared_y_axes().join(axs[3])

    ind = 0
    for algo_kwd in scores1:
        h1 = sns.heatmap(scores1[algo_kwd], cmap="YlGnBu", ax=axs[ind])
        h2 = sns.heatmap(scores2[algo_kwd], cmap="YlGnBu", ax=axs[ind+1])
        
        h1.set_xlabel('Genotyping')
        h2.set_xlabel('Subtyping')

        h2.set_yticks([])

        ind += 2
 
    plt.suptitle(fig_title)
    plt.savefig(fig_file)


if __name__ == "__main__":
 
    """
    ./make_clfs_heatmaps.py 
    ~/Projects/Thesis/dna_bayes_clf/results/viruses/PR01/2019_06/HCV01_CG.json 
    ~/Projects/Thesis/dna_bayes_clf/results/viruses/PR01/2019_06/HCV02_CG.json 
    test_heat.png 
    MultinomNB:Markov 
    test_f1_weighted 
    4:15
    """

    geno_file = sys.argv[1]
    subt_file = sys.argv[2]
    figure_file = sys.argv[3]
    clf_keyword = sys.argv[4]
    metric = sys.argv[5]
    str_k_list = sys.argv[6]

    clfs_list = clf_keyword.split(":")
    se_ks = str_k_list.split(":")

    if len(se_ks) != 2:
        print("K list argument should contain : to separate start and end")
        sys.exit()

    s_klen = int(se_ks[0])
    e_klen = int(se_ks[1])

    k_main_list = list(range(s_klen, e_klen+1))
 
    geno_json = json.load(open(geno_file, "r"))
    subt_json = json.load(open(subt_file, "r"))

    geno_scores = compile_data(geno_json, clfs_list, k_main_list, metric)
    subt_scores = compile_data(subt_json, clfs_list, k_main_list, metric)

    make_figure(geno_scores, subt_scores, k_main_list, metric, figure_file)

