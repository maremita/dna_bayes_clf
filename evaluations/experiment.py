#!/usr/bin/env python

from dna_bayes import seq_collection
from dna_bayes import kmers
from dna_bayes import bayes

import numpy as np
from sklearn.naive_bayes import MultinomialNB

cls_file = "data/viruses/HBV01/class.csv"
seq_file = "data/viruses/HBV01/data.fa"

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
