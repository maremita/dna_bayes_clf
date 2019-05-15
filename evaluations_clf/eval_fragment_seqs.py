#!/usr/bin/env python

import dna_bayes.evaluation as ev 
from dna_bayes import kmers
from dna_bayes import utils
from dna_bayes import seq_collection

from eval_classifiers import eval_clfs

import sys
import json
import os.path
from collections import defaultdict
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from joblib import Parallel, delayed

from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score


__author__ = "amine"


def clf_cross_val(classifier, X_train, X_conc_train, y_train,
        X_test, X_conc_test, y_test, scorer, verbose=True):

    clf, use_X_back, clf_dscp = classifier

    new_clf = clone(clf)

    final_X_train = X_train 
    final_X_test = X_test
    params = {}

    if use_X_back:
        final_X_train = X_conc_train
        final_X_test = X_conc_test
        params = {'v':X_train.shape[1]}
        y_pred = new_clf.fit(final_X_train, y_train, **params).predict(final_X_test)

    else:
        y_pred = new_clf.fit(final_X_train, y_train).predict(final_X_test)

    if verbose: print("Evaluated: {}\r".format(clf_dscp), flush=True)

    score_tmp = clf_dscp, scorer(y_test, y_pred, average="weighted") 

    return score_tmp


def clfs_evaluation_mp(classifiers, X_train, X_conc_train, y_train,
        X_test, X_conc_test, y_test, scorer, n_proc, verbose=True):
  
    clf_cv_scores = Parallel(n_jobs=n_proc, prefer="processes")(
            delayed(clf_cross_val)(classifiers[clf_ind], X_train, 
                X_conc_train, y_train, X_test, X_conc_test, y_test,
                scorer, verbose) for clf_ind in classifiers)

    scores = {clf_dscp:score for clf_dscp, score in clf_cv_scores}

    return scores


def clfs_evaluation(classifiers, X_train, X_conc_train, y_train,
        X_test, X_conc_test, y_test, scorer, verbose=True):

    scores = dict()

    for clf_ind in classifiers:
        classifier, use_X_back, clf_dscp = classifiers[clf_ind]

        new_clf = clone(classifier)

        final_X_train = X_train 
        final_X_test = X_test
        params = {}

        if use_X_back:
            final_X_train = X_conc_train
            final_X_test = X_conc_test
            params = {'v':X_train.shape[1]}
            y_pred = new_clf.fit(final_X_train, y_train, **params).predict(final_X_test)

        else:
            y_pred = new_clf.fit(final_X_train, y_train).predict(final_X_test)

        if verbose: print("Evaluated: {}\r".format(clf_dscp), flush=True)
        
        scores[clf_dscp] = scorer(y_test, y_pred, average="weighted")

    return scores


def clf_evaluation_with_fragments(classifiers, data_seqs, fragments, parents,
        k, full_kmers, nb_iter, scorer, n_proc, random_state=None,
        verbose=False):

    scores_iter = defaultdict(lambda: [0]*nb_iter)
    final_scores = dict()

    if verbose: print("Validation step", flush=True)
 
    ## construct X and X_back dataset
    X_kmer = ev.construct_kmers_data(data_seqs, k, full_kmers=full_kmers)
    X = X_kmer.data
    #print("X shape {}".format(X.shape))
    X_kmers_list = X_kmer.kmers_list
    y = np.asarray(data_seqs.targets)

    # X_back
    X_kmer_back = ev.construct_kmers_data(data_seqs, k-1,
            full_kmers=full_kmers)
    X_back = X_kmer_back.data
    #print("X_back shape {}".format(X_back.shape))
    X_back_list = X_kmer_back.kmers_list

    ## construct fragments dataset
    X_frgmts = kmers.GivenKmersCollection(fragments, X_kmers_list).data
    #print("X_frgmts shape {}".format(X_frgmts.shape))
    X_frgmts_back = kmers.GivenKmersCollection(fragments, X_back_list).data    
    y_frgmts = np.asarray(fragments.targets)
    #print("X_frgmts_back shape {}".format(X_frgmts_back.shape))
    #print("y_frgmts shape {}".format(y_frgmts.shape))

    seq_ind = list(i for i in range(0,len(data_seqs)))
    sss = StratifiedShuffleSplit(n_splits=nb_iter, test_size=0.2, random_state=random_state)

    for ind_iter, (train_ind, test_ind) in enumerate(sss.split(seq_ind, y)):
        if verbose: print("\nIteration {}\n".format(ind_iter), flush=True)

        X_train = X[train_ind]
        X_back_train = X_back[train_ind]
        X_conc_train = np.concatenate((X_train, X_back_train), axis=1)
        y_train = y[train_ind] 

        # Get fragments test indices
        ind_frgmts = np.array([i for p in test_ind for i in parents[p]])
 
        X_test = X_frgmts[ind_frgmts]
        X_back_test = X_frgmts_back[ind_frgmts]
        X_conc_test = np.concatenate((X_test, X_back_test), axis=1)
        y_test = y_frgmts[ind_frgmts]

        #print("X_test shape {}".format(X_test.shape))
        #print("y_test shape {}".format(y_test.shape))
       
        if n_proc == 0 :
            clf_scores = clfs_evaluation(classifiers, X_train, X_conc_train, y_train,
                X_test, X_conc_test, y_test, scorer, verbose=verbose)

        else: 
            clf_scores = clfs_evaluation_mp(classifiers, X_train, X_conc_train, y_train,
                X_test, X_conc_test, y_test, scorer, n_proc, verbose=verbose)

        for clf_dscp in clf_scores:
            scores_iter[clf_dscp][ind_iter] = clf_scores[clf_dscp]

    for clf_dscp in scores_iter:
        scores = np.asarray(scores_iter[clf_dscp]) 
        final_scores[clf_dscp] = [scores.mean(), scores.std()]

    return final_scores


def k_evaluation_with_fragments(seq_file, cls_file, k_main_list, full_kmers,
        frgmt_size, nb_iter, scorer, n_proc, random_state=None, verbose=True):

    k_scores = defaultdict(dict)

    if verbose: print("Dataset construction step", flush=True)

    seq_cv = seq_collection.SeqClassCollection((seq_file, cls_file))

    ## construct fragments dataset
    frgmts_cv = seq_cv.get_fragments(frgmt_size, step=int(frgmt_size/2))
    frgmts_parents = frgmts_cv.get_parents_rank_list()

    classifiers = eval_clfs

    for k_main in k_main_list:
        if verbose: print("\nProcessing k_main={}".format(k_main), flush=True)

        clf_scores = clf_evaluation_with_fragments(classifiers,
                seq_cv, frgmts_cv, frgmts_parents, k_main, full_kmers, 
                nb_iter, scorer, n_proc, random_state=random_state, verbose=verbose)

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

        #axs[ind].bar(kList, means, width, yerr=stds)

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
    plt.show()


if __name__ == "__main__":
 
    """
    ./eval_fragment_seqs.py\ 
    ../data/viruses/HIV01/data.fa\ 
    ../data/viruses/HIV01/class.csv\ 
    ../results/viruses/wcb_2019/tests/HIV01_frg_mp.json\
    4 5 1000 5 4
    """
    
    if len(sys.argv) != 9:
        print("8 arguments are needed!")
        sys.exit()

    print("RUN {}".format(sys.argv[0]), flush=True)

    seq_file = sys.argv[1]
    cls_file = sys.argv[2]
    scores_file = sys.argv[3]
    s_klen=int(sys.argv[4])
    e_klen=int(sys.argv[5])
    fragment_size = int(sys.argv[6])
    nb_iter = int(sys.argv[7])
    n_proc = int(sys.argv[8])

    k_main_list = list(range(s_klen, e_klen+1))
    full_kmers = False
    the_scorer = f1_score

    rs = 0  # random_state
    verbose = True

    if not os.path.isfile(scores_file):
    #if True:
        the_scores = k_evaluation_with_fragments(seq_file, cls_file,
                k_main_list, full_kmers, fragment_size, nb_iter, the_scorer,
                n_proc, random_state=rs, verbose=True)

        with open(scores_file ,"w") as fh_out: 
            json.dump(the_scores, fh_out, indent=2)
 
    else:
       the_scores = json.load(open(scores_file, "r"))

    #pprint(the_scores)
    make_figure(the_scores, k_main_list, scores_file, verbose)
