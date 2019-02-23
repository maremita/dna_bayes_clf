#!/usr/bin/env python

import src.evaluation as ev 
from src import kmers
from src import bayes
from src import utils

import sys
import json
import os.path
from collections import defaultdict
from pprint import pprint

import numpy as np

from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def clf_evaluation_with_fragments(classifiers, data_seqs, k, frgmt_size,
        nb_iter, scorer, random_state=None, verbose=False):

    scores_iter = defaultdict(lambda: [0]*nb_iter)
    final_scores = dict()

    if verbose: print("Validation step")
    
    y = np.asarray(data_seqs.targets)
    
    seq_ind = list(i for i in range(0,len(data_seqs)))
    sss = StratifiedShuffleSplit(n_splits=nb_iter, test_size=0.5)

    # Construct train and test dataset
    for ind_iter, (train_ind, test_ind) in enumerate(sss.split(seq_ind, y)):
        if verbose: print("\nIteration {}\n".format(ind_iter))

        D_train, D_test = data_seqs[train_ind.tolist()], data_seqs[test_ind.tolist()]
        y_train, _ = y[train_ind], y[test_ind]

        # construct X_train
        X_train = kmers.FullKmersCollection(D_train, k=k).data

        # construct fragment dataset (X_test)
        D_test = D_test.get_fragments(frgmt_size, step=int(frgmt_size/2))
        X_test = kmers.FullKmersCollection(D_test, k=k).data
        y_test = np.asarray(D_test.targets)

        for clf_ind in classifiers:
            classifier, options, clf_dscp = classifiers[clf_ind]
 
            new_clf = clone(classifier)
            if verbose: print("Evaluating {}\r".format(clf_dscp))
            y_pred = new_clf.fit(X_train, y_train).predict(X_test)
            
            scores_iter[clf_dscp][ind_iter] = scorer(y_test, y_pred, average="weighted")
       
    for clf_dscp in scores_iter:
        scores = np.asarray(scores_iter[clf_dscp]) 
        final_scores[clf_dscp] = [scores.mean(), scores.std()]

    return final_scores


def k_evaluation_with_fragments(seq_file, cls_file, k_main_list, k_estim,
        frgmt_size, nb_iter, scorer, random_state=None, verbose=True):

    priors="uniform"
    k_scores = dict()
    estim_size = 0.1

    for k_main in k_main_list:
 
        if verbose: print("\nProcessing k_main={}".format(k_main))
        if verbose: print("Dataset construction step")

        seq_cv, seq_estim = ev.construct_split_collection(seq_file, cls_file,
                estim_size, random_state=random_state)

        # Construct the dataset for alpha  estimation
        seq_estim_X = kmers.FullKmersCollection(seq_estim, k=k_estim).data
        seq_estim_y = seq_estim.targets

        if verbose: print("Kmer word dataset construction step")
        all_words_data, all_backs_data = ev.kmer_dataset_construction(k_main, k_estim, alphabet='ACGT')

        if verbose: print("Alpha estimation from sequence dataset for NB Multinomial")
        a_mnom, a_mnom_y = bayes.Bayesian_MultinomialNB.fit_alpha_with_markov(seq_estim_X, seq_estim_y, all_words_data, None)
 
        if verbose: print("Alpha estimation from sequence dataset for Markov Model")
        a_mkov, a_mkov_y = bayes.Bayesian_MarkovModel.fit_alpha_with_markov(seq_estim_X, seq_estim_y, all_words_data, all_backs_data)

        classifiers = {
                0: [bayes.MLE_MultinomialNB(priors=priors), {}, "MLE_MultinomNB"],
                1: [bayes.Bayesian_MultinomialNB(priors=priors, alpha=1e-10), {}, "BAY_MultinomNB_Alpha_1e-10"],
                2: [bayes.Bayesian_MultinomialNB(priors=priors, alpha=1), {}, "BAY_MultinomNB_Alpha_1"],
                3: [bayes.Bayesian_MultinomialNB(priors=priors, alpha=a_mnom, alpha_classes=a_mnom_y), {}, "BAY_MultinomNB_Alpha_estimated"],

                4: [bayes.MLE_MarkovModel(priors=priors), {}, "MLE_Markov"],
                5: [bayes.Bayesian_MarkovModel(priors=priors, alpha=1e-10), {}, "BAY_Markov_Alpha_1e-10"],
                6: [bayes.Bayesian_MarkovModel(priors=priors, alpha=a_mkov, alpha_classes=a_mkov_y), {}, "BAY_Markov_Alpha_estimated"],

                7: [GaussianNB(), {}, "SK_Gaussian_NB"],
                8: [LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=400), {}, "SK_Logistic_Regression"],
                9: [SVC(C=1, kernel="linear"), {}, "SK_Linear_SVC"]
               }

        k_scores[str(k_main)] = clf_evaluation_with_fragments(classifiers,
                seq_cv, k_main, frgmt_size, nb_iter, scorer,
                random_state=random_state, verbose=verbose)

    return k_scores 


if __name__ == "__main__":
 
    """
    ./eval_fragment_seqs.py data/viruses/HPV01/data.fa data/viruses/HPV01/class.csv results/viruses/HPV01_frgmt_250.json
    """

    if len(sys.argv) != 4:
        print("3 arguments are needed!")
        sys.exist()

    seq_file = sys.argv[1]
    cls_file = sys.argv[2]
    scores_file = sys.argv[3]

    k_main_list = list(range(4,10))
    #k_main_list = [4]
    k_estim = 2
    fragment_size = 250
    nb_iter = 5
    the_scorer = f1_score

    rs = 0  # random_state
    verbose = True

    clf_names = { 
            "MLE_MultinomNB":"MLE_MultinomNB",
            "BAY_MultinomNB_Alpha_1e-10":"BAY_MultinomNB_Alpha_1e-10",
            "BAY_MultinomNB_Alpha_1":"BAY_MultinomNB_Alpha_1",
            "BAY_MultinomNB_Alpha_estimated":"BAY_MultinomNB_Alpha_estimated",

            "MLE_Markov":"MLE_Markov",
            "BAY_Markov_Alpha_1e-10":"BAY_Markov_Alpha_1e-10",
            "BAY_Markov_Alpha_estimated":"BAY_Markov_Alpha_estimated",

            "SK_Gaussian_NB":"SK_Gaussian_NB",
            "SK_Logistic_Regression":"SK_Logistic_Regression",
            "SK_Linear_SVC":"SK_Linear_SVC"
            }

    if not os.path.isfile(scores_file):
        the_scores = k_evaluation_with_fragments(seq_file, cls_file,
                k_main_list, k_estim, fragment_size, nb_iter, the_scorer,
                random_state=rs, verbose=True)

        with open(scores_file ,"w") as fh_out: 
            json.dump(the_scores, fh_out, indent=2)
 
    else:
       the_scores = json.load(open(scores_file, "r"))

    the_scores = utils.rearrange_data_struct(the_scores)
    #pprint(the_scores)
    ev.make_figure(the_scores, clf_names, k_main_list, scores_file, verbose)
