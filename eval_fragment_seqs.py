#!/usr/bin/env python

import src.evaluation as ev 
from src import kmers
from src import bayes

import sys
import json
import os.path
from collections import defaultdict
from pprint import pprint

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score


def clf_evaluation_with_fragments(classifiers, data_seqs, nb_iter, scorer,
        random_state=None, verbose=False):

    scores_iter = defaultdict(list)
    final_scores = dict()

    if verbose: print("Validation step")
    
    y = np.asarray(seq_cv.targets)
    
    seq_ind = list(i for i in range(0,len(data_seqs)))
    sss = StratifiedShuffleSplit(n_splits=nb_iter, test_size=0.5)

    # Construct train and test dataset
    for ind_iter, (train_ind, test_ind) in enumerate(sss.split(seq_ind, y)):
        if verbose: print("Iteration {}".format(ind_iter))

        D_train, D_test = data_seqs[train_ind], data_seqs[test_ind]
        y_train, _ = y[train_ind], y[test_ind]

        # construct X_train
        X_train = kmers.FullKmersCollection(D_train, k=k_main).data

        # construct fragment dataset (X_test)
        D_test = D_test.get_fragments(size, step=int(size/2))
        X_test = kmers.FullKmersCollection(D_test, k=k_main).data
        y_test = np.asarray(D_test.targets)
        
        for clf_ind in classifiers:
            classifier, options, clf_dscp = classifiers[clf_ind]

            if verbose: print("Evaluating {}".format(clf_dscp))
            y_pred = classifier.fit(X_train, y_train).predict(X_test)
            
            scores_iter[clf_dscp].append(scorer(y_test, y_pred, average="weighted"))
       
    for clf_dscp in scores_iter:
        scores = np.asarray(scores_iter[clf_dscp]) 
        final_scores[clf_dscp] = [scores.mean(), scores.std()]

    return final_scores


def k_evaluation_with_fragments(seq_file, cls_file, k_main_list, k_estim,
        cv_iter, scorer, random_state=None, verbose=True):

    priors="uniform"
    k_scores = dict()
    estim_size = 0.1

    for k_main in k_main_list:
 
        if verbose: print("\nProcessing k_main={}".format(k_main))
        if verbose: print("Dataset construction step")

        seq_cv, seq_estim = ev.construct_split_collection(seq_file, cls_file, estim_size)

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

        k_scores[str(k_main)] = clf_evaluation_with_fragments(classifiers, seq_cv, cv_iter,
                scorer, random_state=random_state, verbose=verbose)

    return k_scores 


if __name__ == "__main__":
 
    """
    ./eval_fragment_seqs.py data/viruses/HPV01/data.fa data/viruses/HPV01/class.csv results/viruses/HPV01_frgmt.json
    """

    if len(sys.argv) != 4:
        print("3 arguments are needed!")
        sys.exist()

    seq_file = sys.argv[1]
    cls_file = sys.argv[2]
    scores_file = sys.argv[3]

    #k_main_list = list(range(4,10))
    k_main_list = [4]
    k_estim = 2
    rs = 0  # random_state
    verbose = True
    cv_iter = 5
    the_scorer = f1_score

    clf_names = 
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
        the_scores = k_evaluation_with_fragments(seq_file, cls_file, k_main_list, k_estim, 
                cv_iter, the_scorer, random_state=rs,
                verbose=True)

        with open(scores_file ,"w") as fh_out: 
            json.dump(the_scores, fh_out, indent=2)
 
    else:
       the_scores = json.load(open(scores_file, "r"))

    the_scores = utils.rearrange_data_struct(the_scores)
    pprint(the_scores)
    #ev.make_figure(the_scores, clf_names, k_main_list, scores_file, verbose)
