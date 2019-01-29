from kmers import get_kmer_index
from itertools import product
import numpy as np
from scipy.special import logsumexp


class BaseMultinomialNaiveBayes():
    """
    """

    def __init__(self, priors=None):
        self.priors = priors

    def _initial_fit(self, X, targets):
        """
        """
        
        self.classes_ = np.unique(targets)
        n_classes = len(self.classes_)
        v_size = X.shape[1]

        # Compute class priors
        # Calcul des probabilites a priori pour chaque classe

        self.class_counts_ = np.zeros(n_classes)

        for ind in range(n_classes):
            self.class_counts_[ind] = len(targets[targets==self.classes_[ind]])
 
        if self.priors == "uniform":
            self.class_priors_ = np.full(n_classes, 1/n_classes)

        elif self.priors == "ones":
            self.class_priors_ = np.full(n_classes, 1)

        elif self.priors is not None:
            self.class_priors_ = self.priors

        else:
            # P(y_i) = #{y_i} / #{y}
            self.class_priors_ = self.class_counts_ / self.class_counts_.sum()

        # log class priors
        self.log_class_priors_ = np.log(self.class_priors_)

        # compute y per target value
        self.y_ = np.zeros((n_classes, v_size)) 
        self.log_kmer_probs_ = np.zeros(self.y_.shape)

        for ind in range(n_classes):
            X_class = X[targets == self.classes_[ind]]
            # sum word by word
            self.y_[ind, :] = np.sum(X_class, axis=0)

        # compute the sum of ys
        self.Y_ = np.zeros(n_classes)

        #for ind in range(n_classes):
        #    self.Y_[ind] = np.sum(self.y_[ind])
        self.Y_ = self.y_.sum(axis=1)

        return self

    def _log_joint_prob_density(self, X):
        """
        Compute the unnormalized posterior log probability of sequence
 
        I.e. ``log P(C) + log P(sequence | C)`` for all rows x of X, as an array-like of
        shape [n_sequences, n_classes].

        Input is passed to _log_joint_prob_density as-is by predict,
        predict_proba and predict_log_proba. 
        """

        #log_joint_prob_density = []
        #for i in range(n_classes):
        #    # compute the log conditional prob density distribution for class i
        #    log_likelihood_dens = np.dot(X, self.log_kmer_probs_[i])
        #    # compute the log joint prob density distribution for class i
        #    log_joint_prob_density.append(self.log_class_priors_[i] + log_likelihood_dens)
        #log_joint_prob_density = np.array(log_joint_prob_density).T

        return np.dot(X, self.log_kmer_probs_.T) + self.log_class_priors_

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : 

        Returns
        -------
        C : array, shape = (n_sequences)
            Predicted target values for X
        """

        ljb = self._log_joint_prob_density(X)
        # ljb has a shape of (n_sequences, n_classes)

        return self.classes_[np.argmax(ljb, axis=1)]

    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the test vector X.
        
        Parameters
        ----------
        X : 
 
        Returns
        -------
        C : array-like, shape = (n_sequences, n_classes)
            Returns the log-probability of the sequences for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """

        ljb = self._log_joint_prob_density(X)

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

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.
 
        Parameters
        ----------
        X : 

        Returns
        -------
        C : array-like, shape = (n_sequences, n_classes)
            Returns the probability of the sequences for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute `classes_`.
        """

        return np.exp(self.predict_log_proba(X))

class MLE_MultinomialNaiveBayes(BaseMultinomialNaiveBayes):
    """
    """
 
    def fit(self, X, targets):
        self._initial_fit(X, targets)

        # M1
        #for ind in range(n_classes):
        #    self.log_kmer_probs_[ind] = np.log(self.y_[ind]) - np.log(self.Y_[ind])
        #self.log_kmer_probs_ = np.nan_to_num(self.log_kmer_probs_)

        # M2
        #self.log_kmer_probs_ = np.nan_to_num(np.log(self.y_.T) - np.log(self.Y_)).T
        
        # M3
        self.log_kmer_probs_ = np.nan_to_num(np.log(self.y_) - np.log(self.Y_.reshape(-1, 1))) 
 
        return self

class Smooth_MultinomialNaiveBayes(BaseMultinomialNaiveBayes):
    """
    """
    def __init__(self,priors=None, alpha=1e-10):
        super().__init__(priors=priors)
        # validate alpha
        self.alpha = self.check_alpha(alpha)

    def fit(self, X, targets):
        self._initial_fit(X, targets)
        v_size = X.shape[1]

        # Beta
        beta = self.y_ + self.alpha
        beta_sum = self.Y_ + (self.alpha * v_size)

        self.log_kmer_probs_ = np.log(beta) - np.log(beta_sum.reshape(-1, 1))

        return self
 
    # TODO
    def check_alpha(self, alpha):
        return alpha


def alpha_estimate_markov_chain_from(sequences, alphabet, main_k, secd_k):
    v_size = np.power(len(alphabet), main_k)
    prior_alpha = np.zeros(v_size)
    all_kmers = ["".join(t) for t in product(alphabet, repeat=main_k)]

    #

    for i, kmer in enumerate(all_kmers):
        kmer_ind = get_kmer_index(kmer, main_k)

        log_kmer_prior = 0
        p_alpha[kmer_ind] = np.exp(log_kmer_prior)

    return prior_alpha


#    def compute_alpha(self, sequences, alpha):
#        prior_alpha = alpha
#
#        # If alpha is not int, float or array
#        if alpha == "mc_w_p":
#            # Markov chain with probabilities depend on 
#            # the sequence of the kmer 
#            prior_alpha = self._mc_word_dependant_priors()
#        
#        elif alpha == "mc_c_p":
#            # Markov chain with probabilities depend on
#            # the frequence of words per class
#            prior_alpha = self._mc_class_dependant_priors(sequences)
#
#        return prior_alpha
#
#    def _mc_word_dependant_priors(self):
#        """
#        mc for Markov chain
#        """
#        mc_order = 3
#        orders_dict = dict()
#        lw = self.main_k 
#        ly = lw - mc_order + 1
#        lz = lw - mc_order
#
#
#        p_alpha = np.zeros(self.v_size)
#        all_kmers = ["".join(t) for t in product(self.alphabet, repeat=self.main_k)]
#        # get_kmer_index(kmer, k)
#        
#        print("Generate n-order kmers")
#        for i in range(mc_order-1, mc_order+1):
#            orders_dict[i] = kmers.FullKmersCollection(all_kmers, k=i)
#        
#        print("Compute alpha")
#        for i, kmer in enumerate(all_kmers):
#            kmer_ind = get_kmer_index(kmer, lw)
# 
#            logs1 = np.sum([np.log(val) for val in orders_dict[mc_order].data[i] if val != 0])
#            logs2 = np.sum([np.log(val) for val in orders_dict[mc_order-1].data[i] if val != 0])
#            print("logs1 {}\nlogs2 {}\n".format(logs1, logs2))
#
#            log_kmer_prior = (ly * np.log(lz)) - (ly * np.log(lw)) + logs1 - logs2
#            p_alpha[kmer_ind] = np.exp(log_kmer_prior)
#
#        print(p_alpha)
#
#        return 1
#
#    def _mc_class_dependant_priors(self, sequences):
#
#        p_alpha = np.zeros(self.v_size)
#        all_kmers = ["".join(t) for t in product(self.alphabet, repeat=self.v_size)]
#
#        return p_alpha
