#!/usr/bin/env python

from src import seq_collection
from src import kmers
from src import bayes
from src import utils

import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score, StratifiedKFold

# import sys

__author__ = "amine"
__version__ = "0.1"


if __name__ == "__main__":
 
    k_main = 9
    k_estm = 3
    rs = 0  # random_state

    cls_file = "data/viruses/HBV/1_HBV_geo_balanced.csv"
    seq_file = "data/viruses/HBV/1_HBV_geo_balanced.fasta"

    data_seqs = seq_collection.SeqClassCollection((seq_file, cls_file))
 
    # Split sequences to estimation and cv steps
    seq_ind = list(i for i in range(0,len(data_seqs)))
    a, b = next(ShuffleSplit(n_splits=1, test_size=0.1, random_state=rs).split(seq_ind))
 
    seq_train = data_seqs[list(a)]
    seq_estim = data_seqs[list(b)]

    # Learn markov model with dataset estimation
    sq_estim_data = kmers.FullKmersCollection(seq_estim, k=k_estm).data

    # Get kmer word list
    all_words = utils.generate_all_words('ACGT', k_main)
    all_words_data = kmers.FullKmersCollection(all_words, k=k_estm).data

    # Get kmer word for backoff
    all_backs = utils.generate_all_words('ACGT', k_main-1)
    all_backs_data = kmers.FullKmersCollection(all_backs, k=k_estm).data


    # Train a probabilistic classifier with new prior probabilities
    ## Construct the data
    seq_train_data = kmers.FullKmersCollection(seq_train, k=k_main).data
 
    ## Train a Bayesian bayesian classifier
    #bayes_clf1 = bayes.Bayesian_MultinomialNaiveBayes(priors="uniform")
    bayes_clf2 = bayes.Bayesian_MultinomialNaiveBayes(priors="uniform", X_estim=sq_estim_data, y_estim=seq_estim.targets, kmers=all_words_data)

    ## Train a Bayesian Markov classifier
    #markov_clf1 = bayes.Bayesian_MarkovModel(priors="uniform")
    #markov_clf2 = bayes.Bayesian_MarkovModel(priors="uniform", X_estim=sq_estim_data, y_estim=seq_estim.targets, kmers=all_words_data, kmers_backs=all_backs_data)
    skf = StratifiedKFold(n_splits=10, shuffle=False, random_state=None)

    #bayes_sc1 = cross_val_score(bayes_clf1, seq_train_data, seq_train.targets, cv=skf, scoring="f1_weighted")
    bayes_sc2 = cross_val_score(bayes_clf2, seq_train_data, seq_train.targets, cv=skf, scoring="f1_weighted")

    #markov_sc1 = cross_val_score(markov_clf1, seq_train_data, seq_train.targets, cv=skf, scoring="f1_weighted")
    #markov_sc2 = cross_val_score(markov_clf2, seq_train_data, seq_train.targets, cv=skf, scoring="f1_weighted")

    #print(bayes_sc1.mean())
    print(bayes_sc2.mean())
    #print(markov_sc1.mean())
    #print(markov_sc2.mean())
