from collections import defaultdict
import numpy as np
import re


__all__ = ['FullKmersCollection', 'get_kmer_index']


def get_kmer_index(kmer, k):
    """
    Function adapted from module enrich.pyx of
    GenomeClassifier package [Sandberg et al. (2001)]
    """

    f=1
    s=0
    alpha_to_code = {'A':0, 'C':1, 'G':2, 'T':3}

    for i in range(0, k):
        alpha_code=alpha_to_code[kmer[i]]
        s = s + alpha_code * f
        f = f * 4

    return s


class FullKmersCollection(object):

    def __init__(self, sequences, k=5, alphabet="ACGT"):
        self.k = k
        self.alphabet = alphabet
        #
        self.ids = []
        self.v_size = np.power(len(self.alphabet), self.k)
        self.data = np.zeros((len(sequences), self.v_size))
        #
        self._compute_kmers_collection(sequences)

    def _compute_kmers_sequence(self, sequence, ind):
        search = re.compile("^["+self.alphabet+"]+$").search
        
        kmer_array = np.zeros(self.v_size)

        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i + self.k]

            if self.alphabet and bool(search(kmer)) or not self.alphabet:
                ind_kmer = get_kmer_index(kmer, self.k)
                self.data[ind][ind_kmer] += 1

        return self
 
    def _compute_kmers_collection(self, sequences):
        for i, seq in enumerate(sequences):
            self._compute_kmers_sequence(seq.seq._data, i)
            self.ids.append(seq.id)

        return self
