import kmers
from abc import ABC, abstractmethod
import numpy as np


class BaseMultinomialNaiveBayes(ABC):
    """
    """

    def __init__(self, alphabet= "ACGT", priors=None):
        self.priors = priors
        self.alphabet = alphabet

    def _initial_fit(self, sequences, main_k):
        """
        """
        
        self.main_k = main_k
        # ordered list of target labels
        targets = np.asarray(sequences.targets)
        self.classes_ = np.unique(targets)
        self.n_classes = len(self.classes_)
        self.v_size = np.power(len(self.alphabet), self.main_k)

        # Compute class priors
        # Calcul des probabilites a priori pour chaque classe
        # P(y_i) = #{y_i} / #{y}

        self.class_counts_ = np.zeros(self.n_classes)

        for ind in range(self.n_classes):
            self.class_counts_[ind] = len(targets[targets==self.classes_[ind]])
 
        if self.priors == "uniform":
            self.class_priors_ = np.full(self.n_classes, 1/self.n_classes)

        elif self.priors == "ones":
            self.class_priors_ = np.full(self.n_classes, 1)

        elif self.priors is not None:
            self.class_priors_ = self.priors

        else:
            self.class_priors_ = self.class_counts_ / self.class_counts_.sum()

        # compute kmers
        self.main_kmer = kmers.FullKmersCollection(sequences, k=self.main_k, alphabet=self.alphabet)
        #self.secd_kmer = kmers.FullKmersCollection(sequences, k=secd_k, alphabet="ACGT")

        # compute y per target value
        self.y = np.zeros((self.n_classes, self.v_size)) 

        for ind in range(self.n_classes):
            X_class = self.main_kmer.data[targets == self.classes_[ind]]
            # sum word by word
            self.y[ind, :] = np.sum(X_class, axis=0)

        # compute the sum of ys
        self.Y = np.zeros(self.n_classes)

        for ind in range(self.n_classes):
            self.Y[ind] = np.sum(self.y[ind])

        return self

    @abstractmethod
    def _log_joint_prob_density(self, sequences):
        """
        Compute the unnormalized posterior log probability of sequence
        
        I.e. ``log P(C) + log P(sequence | C)`` for all rows x of X, as an array-like of
        shape [n_classes, n_sequences].
        
        Input is passed to _joint_log_likelihood as-is by predict,
        predict_proba and predict_log_proba.
        
        """

    def predict(self, sequences):
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        sequences : 

        Returns
        -------
        C : array, shape = (n_sequences)
            Predicted target values for X
        """

        ljb = self._log_joint_prob_density(sequences)
        # ljb has a shape of (n_sequences, n_classes)

        return self.classes_[np.argmax(ljb, axis=1)]

    def predict_log_proba(self, sequences):
        """
        Return log-probability estimates for the test vector X.
        
        Parameters
        ----------
        sequences : 
 
        Returns
        -------
        C : array-like, shape = (n_sequences, n_classes)
            Returns the log-probability of the sequences for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """

        ljb = self._log_joint_prob_density(sequences)

        # normalize by P(x) = P(f_1, ..., f_n) la formule est fausse
        # We use marginalization to compute P(x)
        # P(x) = sum(P(x, c_i))

        # ljb contains log joint prob densitys for each class
        # We put this values in exp and sum them
        # finally, we compute the log of the sum
        # logsumexp : Compute the log of the sum of exponentials of 
        #             input elements.
        #
        # we substract log_prob_x because we calculate with logs, if we work
        # with densities or probabilities we divide by P(x)

        log_prob_x = logsumexp(ljb, axis=1)

        return ljb - np.atleast_2d(log_prob_x).T

    def predict_proba(self, sequences):
        """
        Return probability estimates for the test vector X.
 
        Parameters
        ----------
        sequences : 

        Returns
        -------
        C : array-like, shape = (n_sequences, n_classes)
            Returns the probability of the sequences for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """

        return np.exp(self.predict_log_proba(sequences))

class MLE_MultinomialNaiveBayes(BaseMultinomialNaiveBayes):
    """
    """
 
    def fit(self, sequences, main_k, secd_k):
        self._initial_fit(sequences, main_k)
        self.prob_kmers = np.zeros(self.y.shape)

        for ind in range(self.n_classes):
            self.prob_kmers[ind] = np.divide(self.y[ind], self.Y[ind]) 

        return self

    def _log_joint_prob_density(self, sequences):
        """
        Compute the unnormalized posterior log probability of sequence
        
        I.e. ``log P(C) + log P(sequence | C)`` for all rows x of X, as an array-like of
        shape [n_classes, n_sequences].
        
        Input is passed to _joint_log_likelihood as-is by predict,
        predict_proba and predict_log_proba.
        
        """
        log_joint_prob_density = []

        seqs_kmer = kmers.FullKmersCollection(sequences, k=self.main_k, alphabet=self.alphabet)
        #
        for i in range(np.size(self.classes_)):
            # compute the log prob of class prior i
            log_prior_i = np.log(self.class_priors_[i])

            # compute the log conditional prob density distribution for class i
            log_likelihood_dens = 0

            # compute the log joint prob density distribution for class i
            log_joint_prob_density.append(log_prior_i + log_likelihood_dens)

        #
        log_joint_prob_density = np.array(log_joint_prob_density).T

        return log_joint_prob_density
