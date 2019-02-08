from .seq_collection import SeqClassCollection
from .utils import get_index_from_kmer
import re
import numpy as np


__all__ = ['FullKmersCollection']


class FullKmersCollection(object):

    def __init__(self, sequences, k=5, alphabet="ACGT"):
        self.k = k
        self.alphabet = alphabet
        #
        self.ids = []
        self.v_size = np.power(len(self.alphabet), self.k)
        self.data = np.zeros((len(sequences), self.v_size))
        #
        if isinstance(sequences, SeqClassCollection):
            self._compute_kmers_from_collection(sequences)

        else:
            self._compute_kmers_from_strings(sequences)

    def _compute_kmers_of_sequence(self, sequence, ind):
        search = re.compile("^["+self.alphabet+"]+$").search
        
        kmer_array = np.zeros(self.v_size)

        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i + self.k]

            if self.alphabet and bool(search(kmer)) or not self.alphabet:
                ind_kmer = get_index_from_kmer(kmer, self.k)
                self.data[ind][ind_kmer] += 1

        return self
 
    def _compute_kmers_from_collection(self, sequences):
        for i, seq in enumerate(sequences):
            self._compute_kmers_of_sequence(seq.seq._data, i)
            self.ids.append(seq.id)

        return self

    def _compute_kmers_from_strings(self, sequences):
        for i, seq in enumerate(sequences):
            self._compute_kmers_of_sequence(seq, i)
            self.ids.append(i)

        return self
