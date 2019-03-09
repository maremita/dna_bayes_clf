#!/usr/bin/env python

from dna_bayes import seq_collection
from dna_bayes import kmers
from dna_bayes import bayes

import numpy as np
from sklearn.naive_bayes import MultinomialNB

cls_file = "data/viruses/HBV01/class.csv"
seq_file = "data/viruses/HBV01/data.fa"

k = 3
alpha = 1e-10

seqco = seq_collection.SeqClassCollection((seq_file, cls_file))
# ordered list of target labels
y = np.asarray(seqco.targets)

k3 = kmers.FullKmersCollection(seqco, k=3)
k2 = kmers.FullKmersCollection(seqco, k=2)

clf = bayes.MLE_MultinomialNaiveBayes()
clf.fit(k2.data, y) 

print('clf.log_kmer_probs_')
print(clf.log_kmer_probs_)

slf = bayes.Bayesian_MultinomialNaiveBayes(alpha=alpha)
slf.fit(k2.data, y)
print('slf.log_kmer_probs_')
print(slf.log_kmer_probs_)

mnb = MultinomialNB(alpha=alpha)
mnb.fit(k2.data, y)
print('mnb.feature_log_prob_')
print(mnb.feature_log_prob_)
