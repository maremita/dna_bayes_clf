from dna_bayes import seq_collection
from dna_bayes import kmers
from dna_bayes import bayes

from scipy import sparse

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


cls_file = "../data/viruses/HBV01/class.csv"
seq_file = "../data/viruses/HBV01/data.fa"

seqco = seq_collection.SeqClassCollection((seq_file, cls_file))

X = kmers.FullKmersCollection(seqco, k=7).data
y = seqco.targets

Xr = sparse.csr_matrix(X)
Xc = sparse.csc_matrix(X)

%timeit np.dot(X[0],X[0])
%timeit Xr[0].dot(Xr[0])


mb = MultinomialNB()
%timeit mb.fit(X,y)
%timeit mb.fit(Xr,y)
%timeit mb.fit(Xc,y)

lr = LogisticRegression()
%timeit lr.fit(X,y)
%timeit lr.fit(Xr,y)
%timeit lr.fit(Xc,y)
