from collections import defaultdict
import numpy as np
import re


__all__ = ['Kmers', 'KmersCollection']


class Kmers(defaultdict):
    """
    """

    def __init__(self, seq, n=1, unit_size=1, alphabet="ACGT"):
        super().__init__(int)
        self.n = n
        self.unit_size = unit_size
        self.mer_size = self.n * self.unit_size
        self.alphabet = alphabet
        self.compute_kmers(seq)

    def compute_kmers(self, sequence):
        search = re.compile("^["+self.alphabet+"]+$").search

        for i in range(len(sequence) - self.mer_size + 1):
            kmer = sequence[i:i + self.mer_size]

            if self.alphabet and bool(search(kmer)) or not self.alphabet:
                self[kmer] += 1


class KmersCollection(object):

    def __init__(self, sequences, unit_size=1, n=1, alphabet="ACGT"):
        self.unit_size = unit_size
        self.n = n
        self.alphabet = alphabet
        #
        self.kmers = defaultdict(lambda: [0] * len(sequences))
        self.ids = []
        self.data = []
        self.kmers_list = []
        #
        self._compute_kmers_collection(sequences)

    def _compute_kmers_collection(self, sequences):
        for i, seq in enumerate(sequences):
            seq_kmers = Kmers(seq.seq._data, n=self.n,
                                unit_size=self.unit_size,
                                alphabet=self.alphabet)

            self.ids.append(seq.id)

            for _kmer in seq_kmers:
                self.kmers[_kmer][i] = seq_kmers[_kmer]

        self.kmers_list = sorted(self.kmers.keys())

        for _kmer in self.kmers_list:
            self.data.append(self.kmers[_kmer])

        self.data = np.array(self.data).transpose()

