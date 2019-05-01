#!/usr/bin/env python

import dna_bayes.evaluation as ev 
from dna_bayes import seq_collection
from dna_bayes import kmers

from eval_classifiers import eval_clfs

import warnings
#warnings.filterwarnings('ignore')

import sys
import json
import os.path
from pprint import pprint
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import StratifiedKFold


__author__ = "amine"


def clfs_validation(classifiers, X, X_b, y, cv_iter, scoring="f1_weighted",
        random_state=None, verbose=False):

    scores = dict()
    X_conc = np.concatenate((X, X_b), axis=1)
 
    if verbose: print("Cross-Validation step", flush=True)


    for clf_ind in classifiers:
        classifier, use_X_back, clf_dscp = classifiers[clf_ind]
        final_X = X
        params = {}

        if use_X_back:
            final_X = X_conc
            params = {'v':X.shape[1]}

        if verbose: print("Evaluating {}".format(clf_dscp), flush=True)

        skf = StratifiedKFold(n_splits=cv_iter, shuffle=True,
                random_state=random_state)

        scores_tmp = cross_val_score(classifier, final_X, y, cv=skf,
                scoring=scoring, fit_params=params)
        scores[clf_dscp] = [scores_tmp.mean(), scores_tmp.std()]

    return scores


def k_evaluation(seq_file, cls_file, k_main_list, full_kmers,
        cv_iter, scoring="f1_weighted", random_state=None, verbose=True):

    k_scores = defaultdict(dict)

    for k_main in k_main_list:
    
        if verbose: print("\nProcessing k_main={}".format(k_main), flush=True)
        if verbose: print("Dataset construction step", flush=True)

        seq_cv = seq_collection.SeqClassCollection((seq_file, cls_file))
 
        # # Data for cross validation
        seq_cv_kmers = ev.construct_kmers_data(seq_cv, k_main,
                full_kmers=full_kmers)
        seq_cv_X = seq_cv_kmers.data
        seq_cv_y = np.asarray(seq_cv.targets)
        #seq_cv_kmers_list = seq_cv_kmers.kmers_list

        seq_cv_back_kmers = ev.construct_kmers_data(seq_cv, k_main-1,
                full_kmers=full_kmers)
        seq_cv_X_back = seq_cv_back_kmers.data
        #seq_cv_back_kmers_list = seq_cv_back_kmers.kmers_list

        classifiers = eval_clfs

        clf_scores = clfs_validation(classifiers, seq_cv_X,
                seq_cv_X_back, seq_cv_y, cv_iter, scoring=scoring, 
                random_state=random_state, verbose=verbose)
        
        for clf_dscp in clf_scores:
            k_scores[clf_dscp][str(k_main)] = clf_scores[clf_dscp]

    return k_scores 


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
 
    plt.suptitle(fig_title)
    plt.savefig(fig_file)
    plt.show()

if __name__ == "__main__":
 
    """
    ./eval_complete_seqs.py data/viruses/HPV01/data.fa data/viruses/HPV01/class.csv results/viruses/HPV01_test.json
    """


    if len(sys.argv) != 7:
        print("6 arguments are needed!")
        sys.exit()

    print("RUN {}".format(sys.argv[0]), flush=True)

    seq_file = sys.argv[1]
    cls_file = sys.argv[2]
    scores_file = sys.argv[3]
    s_klen=int(sys.argv[4])
    e_klen=int(sys.argv[5])
    cv_iter = int(sys.argv[6]) #5

    k_main_list = list(range(s_klen, e_klen+1))
    #k_main_list = [4, 5, 6]
    fullKmers = False
    rs = 0  # random_state
    verbose = True

    if not os.path.isfile(scores_file):
    #if True:
        the_scores = k_evaluation(seq_file, cls_file, k_main_list,
                fullKmers, cv_iter=cv_iter, scoring="f1_weighted",
                random_state=rs, verbose=True)

        with open(scores_file ,"w") as fh_out: 
            json.dump(the_scores, fh_out, indent=2)
 
    else:
       the_scores = json.load(open(scores_file, "r"))

    make_figure(the_scores, k_main_list, scores_file, verbose)
