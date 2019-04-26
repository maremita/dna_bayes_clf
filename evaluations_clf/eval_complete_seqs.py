#!/usr/bin/env python

import dna_bayes.evaluation as ev 
from dna_bayes import bayes
from dna_bayes import seq_collection
from dna_bayes import kmers

import warnings
#warnings.filterwarnings('ignore')

import sys
import json
import os.path
from pprint import pprint
from collections import defaultdict

import numpy as np

from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC


__author__ = "amine"


def clfs_validation(classifiers, X, X_b, y, cv_iter, scoring="f1_weighted",
        random_state=None, verbose=False):

    scores = dict()
    X_conc = np.concatenate((X, X_b), axis=1)
 
    if verbose: print("Cross-Validation step")

    skf = StratifiedKFold(n_splits=cv_iter, shuffle=True,
            random_state=random_state)

    for clf_ind in classifiers:
        classifier, use_X_back, clf_dscp = classifiers[clf_ind]
        final_X = X
        params = {}

        if use_X_back:
            final_X = X_conc
            params = {'v':X.shape[1]}

        if verbose: print("Evaluating {}".format(clf_dscp))

        scores_tmp = cross_val_score(classifier, final_X, y, cv=skf,
                scoring=scoring, fit_params=params)
        scores[clf_dscp] = [scores_tmp.mean(), scores_tmp.std()]

    return scores


def k_evaluation(seq_file, cls_file, k_main_list, full_kmers,
        cv_iter, scoring="f1_weighted", random_state=None, verbose=True):

    priors="uniform"
    k_scores = defaultdict(dict)

    for k_main in k_main_list:
    
        if verbose: print("\nProcessing k_main={}".format(k_main))
        if verbose: print("Dataset construction step")

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

        classifiers = {
                0: [bayes.MLE_MultinomialNB(priors=priors), False ,"MLE_MultinomNB"],
                1: [bayes.Bayesian_MultinomialNB(priors=priors, alpha=1e-10), False, "BAY_MultinomNB_Alpha_1e-10"],
                2: [bayes.Bayesian_MultinomialNB(priors=priors, alpha=1), False,"BAY_MultinomNB_Alpha_1"],

                3: [bayes.MLE_MarkovModel(priors=priors), True, "MLE_Markov"],
                4: [bayes.Bayesian_MarkovModel(priors=priors, alpha=1e-10), True, "BAY_Markov_Alpha_1e-10"],

                5: [GaussianNB(), False, "SK_Gaussian_NB"],
                6: [LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=400), False, "SK_Logistic_Regression"],
                7: [LinearSVC(), False, "SK_Linear_SVC"]
               }

        clf_scores = clfs_validation(classifiers, seq_cv_X,
                seq_cv_X_back, seq_cv_y, cv_iter, scoring=scoring, 
                random_state=random_state, verbose=verbose)
        
        for clf_dscp in clf_scores:
            k_scores[clf_dscp][str(k_main)] = clf_scores[clf_dscp]

        #k_scores[str(k_main)] = clfs_validation(classifiers, seq_cv_X,
        #        seq_cv_X_back, seq_cv_y, cv_iter, scoring=scoring, 
        #        random_state=random_state, verbose=verbose)

    return k_scores 


if __name__ == "__main__":
 
    """
    ./eval_complete_seqs.py data/viruses/HPV01/data.fa data/viruses/HPV01/class.csv results/viruses/HPV01_test.json
    """

    k_main_list = list(range(4,11))
    #k_main_list = [4, 5, 6]
    fullKmers = True
    rs = 0  # random_state
    verbose = True
    cv_iter = 5

    if len(sys.argv) != 4:
        print("3 arguments are needed!")
        sys.exit()

    seq_file = sys.argv[1]
    cls_file = sys.argv[2]
    scores_file = sys.argv[3]

    #if not os.path.isfile(scores_file):
    if True:
        the_scores = k_evaluation(seq_file, cls_file, k_main_list,
                fullKmers, cv_iter=cv_iter, scoring="f1_weighted",
                random_state=rs, verbose=True)

        with open(scores_file ,"w") as fh_out: 
            json.dump(the_scores, fh_out, indent=2)
 
    else:
       the_scores = json.load(open(scores_file, "r"))

    ev.make_figure(the_scores, k_main_list, scores_file, verbose)
