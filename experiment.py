import seq_collection
import kmers
import bayes
from sklearn.naive_bayes import MultinomialNB

cls_file = "data/viruses/HBV/HBV_geo.csv"
seq_file = "data/viruses/HBV/HBV_geo.fasta"

seqco = seq_collection.SeqClassCollection((seq_file, cls_file))

k = 6
alpha = 1e-10

k2 = kmers.FullKmersCollection(seqco, k=k)

clf = bayes.MLE_MultinomialNaiveBayes()
clf.fit(seqco, k) 

print('clf.log_kmer_probs')
print(clf.log_kmer_probs)

slf = bayes.Smooth_MultinomialNaiveBayes(alpha=alpha)
slf.fit(seqco, k)
print('slf.log_kmer_probs')
print(slf.log_kmer_probs)

mnb = MultinomialNB(alpha=alpha)
mnb.fit(k2.data, seqco.targets) 
print('mnb.feature_log_prob_')
print(mnb.feature_log_prob_)
