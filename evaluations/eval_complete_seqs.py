#!/usr/bin/env python

import dna_bayes.evaluation as ev 
from dna_bayes import bayes
from dna_bayes import kmers
from dna_bayes import utils

import warnings
#warnings.filterwarnings('ignore')

import sys
import json
import os.path
from pprint import pprint

import numpy as np

from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


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

        #scores_tmp = cross_validate(classifier, X, y, cv=skf,
        #        scoring=scoring, return_train_score=True)
        #scores[clf_dscp] = utils.ndarrays_tolists(scores_tmp)
        scores_tmp = cross_val_score(classifier, final_X, y, cv=skf,
                scoring=scoring, fit_params=params)
        scores[clf_dscp] = [scores_tmp.mean(), scores_tmp.std()]

    return scores


def k_evaluation(seq_file, cls_file, k_main_list, k_estim, full_kmers,
        cv_iter, scoring="f1_weighted", random_state=None, verbose=True):

    priors="uniform"
    k_scores = dict()
    estim_size = 0.1

    for k_main in k_main_list:
    
        if verbose: print("\nProcessing k_main={}".format(k_main))
        if verbose: print("Dataset construction step")

        seq_cv, seq_estim = ev.construct_split_collection(seq_file, cls_file,
                estim_size, random_state=random_state)
        
        # # Data for cross validation
        seq_cv_kmers = ev.construct_kmers_data(seq_cv, k_main,
                full_kmers=full_kmers)
        seq_cv_X = seq_cv_kmers.data
        seq_cv_y = np.asarray(seq_cv.targets)
        seq_cv_kmers_list = seq_cv_kmers.kmers_list

        seq_cv_back_kmers = ev.construct_kmers_data(seq_cv, k_main-1,
                full_kmers=full_kmers)
        seq_cv_X_back = seq_cv_back_kmers.data
        seq_cv_back_kmers_list = seq_cv_back_kmers.kmers_list

        # # Data for alpha estimation
        seq_estim_kmers = ev.construct_kmers_data(seq_estim, k_estim,
                full_kmers=full_kmers)
        seq_estim_X = seq_estim_kmers.data
        seq_estim_y = np.asarray(seq_estim.targets)
        seq_estim_kmers_list = seq_estim_kmers.kmers_list

        seq_estim_kmers_back = ev.construct_kmers_data(seq_estim, k_estim-1,
                full_kmers=full_kmers)
        seq_estim_X_back = seq_estim_kmers_back.data
        seq_estim_back_kmers_list = seq_estim_kmers_back.kmers_list

        X_estim = np.concatenate((seq_estim_X, seq_estim_X_back), axis=1)

        if verbose: print("Kmer word dataset construction step")

        words_data = kmers.GivenKmersCollection(seq_cv_kmers_list,
                seq_estim_kmers_list).data
        words_data_back = kmers.GivenKmersCollection(seq_cv_kmers_list,
                seq_estim_back_kmers_list).data
        
        X_words = np.concatenate((words_data, words_data_back), axis=1)

        backs_data = kmers.GivenKmersCollection(seq_cv_back_kmers_list,
                seq_estim_kmers_list).data
        backs_data_back = kmers.GivenKmersCollection(seq_cv_back_kmers_list,
                seq_estim_back_kmers_list).data

        X_backs = np.concatenate((backs_data, backs_data_back), axis=1)

        if verbose: print("Alpha estimation from "
                "sequence dataset for NB Multinomial")
        a_mnom, a_mnom_y = \
                bayes.Bayesian_MultinomialNB.fit_alpha_with_markov(
                        X_estim, seq_estim_y, X_words, seq_estim_X.shape[1])
 
        if verbose: print("Alpha estimation from "
                "sequence dataset for Markov Model")
        a_mkov, a_mkov_y = \
                bayes.Bayesian_MarkovModel.fit_alpha_with_markov(
                        X_estim, seq_estim_y, X_words, X_backs,
                        seq_estim_X.shape[1])

        classifiers = {
                0: [bayes.MLE_MultinomialNB(priors=priors), False ,"MLE_MultinomNB"],
                1: [bayes.Bayesian_MultinomialNB(priors=priors, alpha=1e-10), False, "BAY_MultinomNB_Alpha_1e-10"],
                2: [bayes.Bayesian_MultinomialNB(priors=priors, alpha=1), False,"BAY_MultinomNB_Alpha_1"],
                3: [bayes.Bayesian_MultinomialNB(priors=priors, alpha=a_mnom, alpha_classes=a_mnom_y), False, "BAY_MultinomNB_Alpha_estimated"],

                4: [bayes.MLE_MarkovModel(priors=priors), True, "MLE_Markov"],
                5: [bayes.Bayesian_MarkovModel(priors=priors, alpha=1e-10), True, "BAY_Markov_Alpha_1e-10"],
                6: [bayes.Bayesian_MarkovModel(priors=priors, alpha=a_mkov, alpha_classes=a_mkov_y), True, "BAY_Markov_Alpha_estimated"],

                7: [GaussianNB(), False, "SK_Gaussian_NB"],
                8: [LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=400), False, "SK_Logistic_Regression"],
                9: [SVC(C=1, kernel="linear"), False, "SK_Linear_SVC"]
               }

        k_scores[str(k_main)] = clfs_validation(classifiers, seq_cv_X,
                seq_cv_X_back, seq_cv_y, cv_iter, scoring=scoring, 
                random_state=random_state, verbose=verbose)

    return k_scores 


if __name__ == "__main__":
 
    """
    ./eval_complete_seqs.py data/viruses/HPV01/data.fa data/viruses/HPV01/class.csv results/viruses/HPV01_test.json
    """

    k_main_list = list(range(4,10))
    #k_main_list = [4, 5, 6]
    k_estim = 3
    fullKmers = False
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
        the_scores = k_evaluation(seq_file, cls_file, k_main_list, k_estim,
                fullKmers, cv_iter=cv_iter, scoring="f1_weighted",
                random_state=rs, verbose=True)

        with open(scores_file ,"w") as fh_out: 
            json.dump(the_scores, fh_out, indent=2)
 
    else:
       the_scores = json.load(open(scores_file, "r"))

    the_scores = utils.rearrange_data_struct(the_scores)
    ev.make_figure(the_scores, k_main_list, scores_file, verbose)
