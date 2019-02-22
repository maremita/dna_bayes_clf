#!/usr/bin/env python

import src.evaluation as ev 
from src import bayes
from src import utils

import sys
import json
import os.path
from pprint import pprint

from sklearn.model_selection import cross_validate, cross_val_score, StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler

__author__ = "amine"
__version__ = "0.3"


def clf_evaluation(classifiers, X, y, cv_iter, scoring="f1_weighted",
        random_state=None, verbose=False):

    scores = dict()
 
    if verbose: print("Cross-Validation step")

    skf = StratifiedKFold(n_splits=cv_iter, shuffle=True, random_state=random_state)
    
    #if verbose: print("Scaling X")
    #X_scaled = MinMaxScaler(feature_range=(0, 1), copy=True).fit_transform(X)

    for clf_ind in classifiers:
        classifier, options, clf_dscp = classifiers[clf_ind]
        the_X = X

        #if 'scale_X' in options and options['scale_X'] == True:
        #    the_X = X_scaled

        if verbose: print("Evaluating {}".format(clf_dscp))

        #scores_tmp = cross_validate(classifier, the_X, y, cv=skf, scoring=scoring, return_train_score=True)
        #scores[clf_dscp] = utils.ndarrays_tolists(scores_tmp)
        scores_tmp = cross_val_score(classifier, the_X, y, cv=skf, scoring=scoring)
        scores[clf_dscp] = [scores_tmp.mean(), scores_tmp.std()]

    return scores


def k_evaluation(seq_file, cls_file, k_main_list, k_estim,
        cv_iter, scoring="f1_weighted", random_state=None, verbose=True):

    priors="uniform"
    k_scores = dict()
    for k_main in k_main_list:
    
        if verbose: print("\nProcessing k_main={}".format(k_main))
        if verbose: print("Dataset construction step")
        seq_cv_X, seq_cv_y, seq_estim_X, seq_estim_y = ev.seq_dataset_construction(seq_file, cls_file, 0.1, k_main, k_estim)

        if verbose: print("Kmer word dataset construction step")
        all_words_data, all_backs_data = ev.kmer_dataset_construction(k_main, k_estim, alphabet='ACGT')

        if verbose: print("Alpha estimation from sequence dataset for NB Multinomial")
        a_mnom, a_mnom_y = bayes.Bayesian_MultinomialNB.fit_alpha_with_markov(seq_estim_X, seq_estim_y, all_words_data, None)
 
        if verbose: print("Alpha estimation from sequence dataset for Markov Model")
        a_mkov, a_mkov_y = bayes.Bayesian_MarkovModel.fit_alpha_with_markov(seq_estim_X, seq_estim_y, all_words_data, all_backs_data)

        classifiers = {
                0: [bayes.MLE_MultinomialNB(priors=priors), {}, "MLE Multinomial NBayes"],
                1: [bayes.Bayesian_MultinomialNB(priors=priors, alpha=1e-10), {'scale_X':False}, "Bayesian Multinomial NBayes with alpha=1e-10"],
                2: [bayes.Bayesian_MultinomialNB(priors=priors, alpha=1), {}, "Bayesian Multinomial NBayes with alpha=1"],
                3: [bayes.Bayesian_MultinomialNB(priors=priors, alpha=a_mnom, alpha_classes=a_mnom_y), {}, "Bayesian Multinomial NBayes with estimated alpha"],

                4: [bayes.MLE_MarkovModel(priors=priors), {}, "MLE Markov model"],
                5: [bayes.Bayesian_MarkovModel(priors=priors, alpha=1e-10), {}, "Bayesian Markov model with alpha=1e-10"],
                6: [bayes.Bayesian_MarkovModel(priors=priors, alpha=a_mkov, alpha_classes=a_mkov_y), {}, "Bayesian Markov model with estimated alpha"],

                7: [GaussianNB(), {'scale_X':False}, "Gaussian Naive Bayes"],
                8: [LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=400), {'scale_X':False}, "Logistic Regression"],
                9: [SVC(C=1, kernel="linear"), {'scale_X':False}, "Linear SVC"]
               }

        k_scores[str(k_main)] = clf_evaluation(classifiers, seq_cv_X, seq_cv_y, cv_iter,
                scoring=scoring, random_state=random_state, verbose=verbose)

    return k_scores 


if __name__ == "__main__":
 
    """
    ./eval_complete_seqs.py data/viruses/HPV01/data.fa data/viruses/HPV01/class.csv results/viruses/HPV01.json
    """

    #k_main_list = list(range(4,10))
    k_main_list = [4,5,6]
    k_estim = 2
    rs = 0  # random_state
    verbose = True
    cv_iter = 5

    if len(sys.argv) != 4:
        print("3 arguments are needed!")
        sys.exist()

    seq_file = sys.argv[1]
    cls_file = sys.argv[2]
    scores_file = sys.argv[3]

    clf_names = {
            "MLE Multinomial NBayes":"MLE_MultinomNB",
            "Bayesian Multinomial NBayes with alpha=1e-10":"BAY_MultinomNB_Alpha_1e-10",
            "Bayesian Multinomial NBayes with alpha=1":"BAY_MultinomNB_Alpha_1",
            "Bayesian Multinomial NBayes with estimated alpha":"BAY_MultinomNB_Alpha_estimated",

            "MLE Markov model":"MLE_Markov",
            "Bayesian Markov model with alpha=1e-10":"BAY_Markov_Alpha_1e-10",
            "Bayesian Markov model with estimated alpha":"BAY_Markov_Alpha_estimated",

            "Gaussian Naive Bayes":"SK_Gaussian_NB",
            "Logistic Regression":"SK_Logistic_Regression",
            "Linear SVC":"SK_Linear_SVC"
            }

    if not os.path.isfile(scores_file):
        the_scores = k_evaluation(seq_file, cls_file, k_main_list, k_estim, 
                cv_iter=cv_iter, scoring="f1_weighted", random_state=rs,
                verbose=True)

        with open(scores_file ,"w") as fh_out: 
            json.dump(the_scores, fh_out, indent=2)
 
    else:
       the_scores = json.load(open(scores_file, "r"))

    the_scores = utils.rearrange_data_struct(the_scores)
    ev.make_figure(the_scores, clf_names, k_main_list, scores_file, verbose)

