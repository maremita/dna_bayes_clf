from .utils import get_index_from_kmer

import re
from collections import defaultdict
from itertools import product
from .seq_collection import SeqClassCollection

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
        self.kmers_list = ["".join(t) for t in product(alphabet, repeat=k)]
        #
        if isinstance(sequences, SeqClassCollection):
            self._compute_kmers_from_collection(sequences)

        else:
            self._compute_kmers_from_strings(sequences)

    def _compute_kmers_of_sequence(self, sequence, ind):
        search = re.compile("^["+self.alphabet+"]+$").search
        
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


class SeenKmersCollection(object):

    def __init__(self, sequences, k=5, alphabet="ACGT"):
        self.k = k
        self.alphabet = alphabet
        #
        self.ids = []
        self.v_size = 0
        self.data = []
        self.dict_data = defaultdict(lambda: [0]*len(sequences))
        self.kmers_list = []

        if isinstance(sequences, SeqClassCollection):
            self._compute_kmers_from_collection(sequences)

        else:
            self._compute_kmers_from_strings(sequences)

    def _compute_kmers_of_sequence(self, sequence, ind):
        search = re.compile("^["+self.alphabet+"]+$").search
 
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i + self.k]

            if self.alphabet and bool(search(kmer)) or not self.alphabet:
                self.dict_data[kmer][ind] += 1

        return self

    def _compute_kmers_from_collection(self, sequences):
        for i, seq in enumerate(sequences):
            self._compute_kmers_of_sequence(seq.seq._data, i)
            self.ids.append(seq.id)
        
        self._construct_data()

        return self

    def _compute_kmers_from_strings(self, sequences):
        for i, seq in enumerate(sequences):
            self._compute_kmers_of_sequence(seq, i)
            self.ids.append(i)

        self._construct_data()

        return self

    def _construct_data(self):
        # Get Kmers list
        self.kmers_list = self.dict_data.keys()
        self.v_size = len(self.kmers_list)

        # Convert to numpy
        self.data = np.array([ self.dict_data[x] for x in self.dict_data ]).T

        return self

