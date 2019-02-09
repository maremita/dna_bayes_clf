#!/usr/bin/env python

from lib import seq_collection
from lib import kmers
from lib import bayes
from lib import utils

import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score, StratifiedKFold

# import sys

__author__ = "amine"
__version__ = "0.1"


if __name__ == "__main__":
    
    k_main = 7
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
    markov_model = bayes.Bayesian_MarkovModel(priors="ones").fit(sq_estim_data, seq_estim.targets)

    # Get kmer word list
    all_words = utils.generate_all_words('ACGT', k_main)
    all_words_data = kmers.FullKmersCollection(all_words, k=k_estm).data

    # Get kmer word for backoff
    all_backs = utils.generate_all_words('ACGT', k_main-1)
    all_backs_data = kmers.FullKmersCollection(all_backs, k=k_estm).data

    # get probabilities of kmer words
    prob_kmers = markov_model.predict_proba(all_words_data)
    ##  Normalization
    prob_kmers = prob_kmers/prob_kmers.sum(axis=0)
    prob_kmers = np.transpose(prob_kmers)
    #print(prob_kmers)

    # get probabilities of kmer words backoffs
    prob_backs = markov_model.predict_proba(all_backs_data)
    ##  Normalization
    prob_backs = prob_backs/prob_backs.sum(axis=0)
    prob_backs = np.transpose(prob_backs)

    # construct alpha for a Markov classifier
    alpha_mrkv = (prob_kmers, prob_backs) 

    # Train a probabilistic classifier with new prior probabilities
    ## Construct the data
    seq_train_data = kmers.FullKmersCollection(seq_train, k=k_main).data
    
    ## Train a Bayesian bayesian classifier
    bayes_clf = bayes.Bayesian_MultinomialNaiveBayes(priors="uniform", alpha=prob_kmers)

    ## Train a Bayesian Markov classifier
    markov_clf = bayes.Bayesian_MarkovModel(priors="uniform", alpha=alpha_mrkv)

    skf = StratifiedKFold(n_splits=10, shuffle=False, random_state=None)

    bayes_sc = cross_val_score(bayes_clf, seq_train_data, seq_train.targets, cv=skf, scoring="f1_weighted")
    markov_sc = cross_val_score(markov_clf, seq_train_data, seq_train.targets, cv=skf, scoring="f1_weighted")

    print(bayes_sc.mean())
    print(markov_sc.mean())
