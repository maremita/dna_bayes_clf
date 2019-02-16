#!/usr/bin/env python

from src import seq_collection
from src import kmers
from src import bayes
from src import utils

import json
from pprint import pprint

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_validate, StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


__author__ = "amine"
__version__ = "0.1"


def seq_dataset_construction(seq_file, cls_file, k_main=9, k_estim=3,
        random_state=None, verbose=False):

    data_seqs = seq_collection.SeqClassCollection((seq_file, cls_file))

    # Split sequences for estimation and cv steps
    seq_ind = list(i for i in range(0,len(data_seqs)))
    a, b = next(StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state).split(seq_ind, data_seqs.targets))

    seq_cv = data_seqs[list(a)]
    seq_estim = data_seqs[list(b)]

    # Construct the dataset for alpha  estimation
    seq_estim_data = kmers.FullKmersCollection(seq_estim, k=k_estim).data
    seq_estim_targets = seq_estim.targets

    # Construct the data for cross-validation
    seq_cv_data = kmers.FullKmersCollection(seq_cv, k=k_main).data
    seq_cv_targets = seq_cv.targets

    return seq_cv_data, seq_cv_targets, seq_estim_data, seq_estim_targets


def kmer_dataset_construction(k_main=9, k_estim=3,
        alphabet='ACGT', verbose=False):

    # Get kmer word list
    all_words = utils.generate_all_words(alphabet, k_main)
    all_words_data = kmers.FullKmersCollection(all_words, k=k_estim).data

    # Get kmer word for backoff
    all_backs = utils.generate_all_words(alphabet, k_main-1)
    all_backs_data = kmers.FullKmersCollection(all_backs, k=k_estim).data

    return all_words_data, all_backs_data


def clf_evaluation(classifiers, X, y, cv_iter=5, scoring=["f1_weighted"],
        random_state=None, verbose=False):

    scores = dict()
    
    if verbose: print("\nCross-Validation step")

    skf = StratifiedKFold(n_splits=cv_iter, shuffle=True, random_state=random_state)

    for clf_ind in classifiers:
        classifier, params, clf_dscp = classifiers[clf_ind]
        clf = classifier(**params)

        if verbose: print("\nEvaluating {}".format(clf_dscp))

        scores_tmp = cross_validate(clf, X, y, cv=skf, scoring=scoring, return_train_score=True)
        scores[clf_dscp] = ndarrays_tolists(scores_tmp)

    return scores

def ndarrays_tolists(obj):
    new_obj = dict()

    for key in obj:
        if isinstance(obj[key], np.ndarray):
            new_obj[key] = obj[key].tolist()

        else:
            new_obj[key] = obj[key]

    return new_obj


if __name__ == "__main__":

    k_main = 9
    k_estim = 3
    rs = 0  # random_state
    verbose = True
    priors="uniform"
 
    scores_file = "results/viruses/HIV02.json"
    cls_file = "data/viruses/HIV02/class.csv"
    seq_file = "data/viruses/HIV02/data.fa"

    if verbose: print("\nDataset construction step")
    seq_cv_X, seq_cv_y, seq_estim_X, seq_estim_y = seq_dataset_construction(seq_file, cls_file, k_main, k_estim)

    if verbose: print("\nKmer word dataset construction step")
    all_words_data, all_backs_data = kmer_dataset_construction(k_main, k_estim)

    if verbose: print("\nAlpha estimation from sequence dataset for NB Multinomial")
    a_mnom, a_mnom_y = bayes.Bayesian_MultinomialNB.fit_alpha_with_markov(seq_estim_X, seq_estim_y, all_words_data, None)
 
    if verbose: print("\nAlpha estimation from sequence dataset for Markov Model")
    a_mkov, a_mkov_y = bayes.Bayesian_MarkovModel.fit_alpha_with_markov(seq_estim_X, seq_estim_y, all_words_data, all_backs_data)

    classifiers = {
            0: [bayes.MLE_MultinomialNB, {'priors':priors}, "MLE Multinom NBayes"],
            1: [bayes.Bayesian_MultinomialNB, {'priors':priors, 'alpha':1e-10}, "Bayesian Multinom NBayes with alpha=1e-10"],
            2: [bayes.Bayesian_MultinomialNB, {'priors':priors, 'alpha':1}, "Bayesian Multinom NBayes with alpha=1"],
            3: [bayes.Bayesian_MultinomialNB, {'priors':priors, 'alpha':a_mnom, 'alpha_classes':a_mnom_y}, "Bayesian Multinom NBayes with estimated alpha"],

            4: [bayes.MLE_MarkovModel, {'priors':priors}, "MLE Markov model"],
            5: [bayes.Bayesian_MarkovModel, {'priors':priors, 'alpha':1e-10}, "Bayesian Markov model with alpha=1e-10"],
            6: [bayes.Bayesian_MarkovModel, {'priors':priors, 'alpha':a_mkov, 'alpha_classes':a_mkov_y}, "Bayesian Markov model with estimated alpha"],

            7: [GaussianNB, {}, "Gaussian Naibe Bayes"],
            8: [LogisticRegression, {'multi_class':'multinomial', 'solver':'lbfgs', 'max_iter':100}, "Logistic Regression"],
            9: [SVC, {'C':1, 'kernel':"linear"}, "Linear SVC"]
           }

    scores = clf_evaluation(classifiers, seq_cv_X, seq_cv_y, cv_iter=5,
            scoring=["f1_weighted"], random_state=rs, verbose=verbose)

    #print(scores)

    with open(scores_file ,"w") as fh_out: 
        json.dump(scores, fh_out, indent=2) 
