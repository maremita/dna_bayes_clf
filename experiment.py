#!/usr/bin/env python

from src import seq_collection
from src import kmers
from src import bayes

import numpy as np
from sklearn.naive_bayes import MultinomialNB

cls_file = "data/viruses/HBV/HBV_geo.csv"
seq_file = "data/viruses/HBV/HBV_geo.fasta"

k = 6
alpha = 1e-10

seqco = seq_collection.SeqClassCollection((seq_file, cls_file))
# ordered list of target labels
targets = np.asarray(seqco.targets)

k2 = kmers.FullKmersCollection(seqco, k=k)

clf = bayes.MLE_MultinomialNaiveBayes()
clf.fit(k2.data, targets) 

print('clf.log_kmer_probs_')
print(clf.log_kmer_probs_)

slf = bayes.Bayesian_MultinomialNaiveBayes(alpha=alpha)
slf.fit(k2.data, targets)
print('slf.log_kmer_probs_')
print(slf.log_kmer_probs_)

mnb = MultinomialNB(alpha=alpha)
mnb.fit(k2.data, targets)
print('mnb.feature_log_prob_')
print(mnb.feature_log_prob_)
