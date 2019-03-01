#!/usr/bin/env python

import dna_bayes.evaluation as ev
from dna_bayes import bayes
from dna_bayes import utils

import warnings
warnings.filterwarnings('ignore')

import sys
import json
import os.path
from collections import defaultdict
from pprint import pprint

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


def clfs_validation_holdout(classifiers, X, y, test_size, scorer,
        random_state=None, verbose=True):

    scores = dict()

    #X_train, X_test, y_train, y_test = train_test_split(
    #        X, y, test_size=test_size, random_state=random_state)

    a, b = next(StratifiedShuffleSplit(n_splits=1, test_size=test_size,
        random_state=random_state).split(X, y))
    
    X_train = X[a]; y_train = y[a]
    X_test = X[b]; y_test = y[b]

    for clf_ind in classifiers:
        classifier, clf_dscp = classifiers[clf_ind]

        if verbose: print("Evaluating {}".format(clf_dscp))

        new_clf = clone(classifier)
        new_clf.fit(X_train, y_train)
        y_pred = new_clf.predict(X_test)

        score_tmp =  scorer(y_test, y_pred, average="weighted")
        scores[clf_dscp] = score_tmp

    return scores


def k_evaluation(seq_file, cls_file, k_main_list, k_estim_list,
        r_estim_list, nb_iter, scorer, random_state=None,
        verbose=True):

    priors="uniform"
    k_scores = defaultdict(dict)

    classifiers = {
        #0: [bayes.MLE_MultinomialNB(priors=priors), "MLE_MNB"],
        0: [bayes.Bayesian_MultinomialNB(priors=priors,
            alpha=1e-10), "BAY_MNB_A_1e-10"],
        1: [bayes.Bayesian_MultinomialNB(priors=priors,
            alpha=1), "BAY_MNB_A_1"],
        #2: [LogisticRegression(multi_class='multinomial',
        #    solver='lbfgs', max_iter=400),
        #    "SK_Logistic_Regression"]
        }
    len_clfs = len(classifiers)
    

    for r_estim in r_estim_list:

        if verbose:
            print("\nProcessing r_estim={}".format(r_estim))

        # initialization
        for ind in classifiers:
            _, clf_dscp = classifiers[ind]
            k_scores[str(r_estim)][clf_dscp] = defaultdict(dict)

        for k in k_estim_list:
            clf_dscp = "BAY_MNB_A_estim_k{}".format(k)
            k_scores[str(r_estim)][clf_dscp] = defaultdict(dict)

        for k_main in k_main_list:
            if verbose:
                print("Processing k_main={}\n".format(k_main))

            scores_iter = defaultdict(list)
            #k_scores[str(r_estim)][str(k_main)] = defaultdict(dict)
            
            for ind_iter in range(nb_iter):

                if verbose:
                    print("ind_iter={}".format(ind_iter), end="\r")

                seq_cv, seq_estim = ev.construct_split_collection(
                        seq_file, cls_file, r_estim, None)

                seq_cv_X, seq_cv_y = ev.construct_Xy_data(seq_cv, k_main)
               
                # Estimate Alpha for MNB classifiers depending on k_estim
                for k_i, k_estim in enumerate(k_estim_list):
                    #print("Processing k_estim={}".format(k_estim))

                    seq_estim_X, seq_estim_y = ev.construct_Xy_data(
                            seq_estim, k_estim)

                    all_words_data, _ = ev.kmer_dataset_construction(
                            k_main, k_estim, alphabet='ACGT')

                    a_mnom, a_mnom_y = bayes.Bayesian_MultinomialNB.\
                            fit_alpha_with_markov(seq_estim_X, seq_estim_y,
                                    all_words_data, None)
                    
                    clf_ind = len_clfs + k_i

                    classifiers[clf_ind] = [bayes.Bayesian_MultinomialNB(
                        priors=priors, alpha=a_mnom, 
                        alpha_classes=a_mnom_y), 
                        "BAY_MNB_A_estim_k{}".format(k_estim)]

                clfs_scores = clfs_validation_holdout(classifiers,
                        seq_cv_X, seq_cv_y, 0.5, scorer,
                        random_state=random_state, verbose=False) 

                for clf_dscp in clfs_scores:
                    scores_iter[clf_dscp].append(clfs_scores[clf_dscp])

            #pprint(scores_iter)
            for clf_dscp in scores_iter:
                scores = np.asarray(scores_iter[clf_dscp]) 
                k_scores[str(r_estim)][clf_dscp][str(k_main)] = [
                        scores.mean(), scores.std()]

    return k_scores


if __name__ == "__main__":

    """
    ./eval_estimation_model.py data/viruses/HPV01/data.fa \
    data/viruses/HPV01/class.csv \
    results/viruses/20190225_eval_estimation/HPV01.json
    """

    k_main_list = list(range(4,10))
    #k_main_list = [7, 8]
    k_estim_list = [2, 3]
    r_estim_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    #r_estim_list = [0.1, 0.2]
    rs = 0  # random_state
    verbose = True
    nb_iter = 10
    the_scorer = f1_score

    if len(sys.argv) != 4:
        print("3 arguments are needed!")
        sys.exit()

    seq_file = sys.argv[1]
    cls_file = sys.argv[2]
    scores_file = sys.argv[3]

    if not os.path.isfile(scores_file):
    #if True:
        the_scores = k_evaluation(seq_file, cls_file, k_main_list,
                k_estim_list, r_estim_list, nb_iter, 
                the_scorer, random_state=rs, verbose=True)

        with open(scores_file ,"w") as fh_out:
            json.dump(the_scores, fh_out, indent=2)

    else:
       the_scores = json.load(open(scores_file, "r"))

    #pprint(the_scores)
    #the_scores = utils.rearrange_data_struct(the_scores)
    ev.make_figure2(the_scores, k_main_list, scores_file, verbose)

