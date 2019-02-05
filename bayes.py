from utils import compute_backoff_words, check_alpha

from abc import ABC, abstractmethod
import numpy as np
from scipy.special import logsumexp


class BaseBayesClassification(ABC):
    """
    """

    def __init__(self, priors=None):
        self.priors = priors

    def _class_prior_fit(self, X, targets):
        """
        """
        
        self.classes_ = np.unique(targets)
        self.n_classes_ = len(self.classes_)

        # Compute class priors
        # Calcul des probabilites a priori pour chaque classe

        self.class_counts_ = np.zeros(self.n_classes_)

        for ind in range(self.n_classes_):
            self.class_counts_[ind] = len(targets[targets==self.classes_[ind]])
 
        if self.priors == "uniform":
            self.class_priors_ = np.full(self.n_classes_, 1/self.n_classes_)

        elif self.priors == "ones":
            self.class_priors_ = np.full(self.n_classes_, 1)

        elif self.priors is not None:
            self.class_priors_ = self.priors

        else:
            self.class_priors_ = self.class_counts_ / self.class_counts_.sum()

        # log class priors
        self.log_class_priors_ = np.log(self.class_priors_)

        return self
    
    @abstractmethod
    def _log_joint_prob_density(self, X):
        """
        Compute the unnormalized posterior log probability of sequence
 
        I.e. ``log P(C) + log P(sequence | C)`` for all rows x of X, as an array-like of
        shape [n_sequences, n_classes].

        Input is passed to _log_joint_prob_density as-is by predict,
        predict_proba and predict_log_proba. 
        """

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


class BaseMultinomialNaiveBayes(BaseBayesClassification):
    """
    """

    def __init__(self, priors=None):
        self.priors = priors

    def _initial_fit(self, X, targets):
        """
        """

        # fit the priors
        self._class_prior_fit(X, targets)
        
        self.v_size_ = X.shape[1]


        # compute y per target value
        self.y_ = np.zeros((self.n_classes_, self.v_size_)) 
        self.log_kmer_probs_ = np.zeros(self.y_.shape)

        for ind in range(self.n_classes_):
            X_class = X[targets == self.classes_[ind]]
            # sum word by word
            self.y_[ind, :] = np.sum(X_class, axis=0)

        # compute the sum of ys
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


class Bayesian_MultinomialNaiveBayes(BaseMultinomialNaiveBayes):
    """
    """
    def __init__(self, priors=None, alpha=1e-10):
        super().__init__(priors=priors)
        # validate alpha
        self.alpha = check_alpha(alpha)

    def fit(self, X, targets):
        self._initial_fit(X, targets)

        # Beta
        beta = self.y_ + self.alpha
        beta_sum = self.Y_ + (self.alpha * self.v_size_)

        self.log_kmer_probs_ = np.log(beta) - np.log(beta_sum.reshape(-1, 1))

        return self
 

class BaseMarkovModel(BaseBayesClassification):
    """
    """

    def __init__(self, priors=None):
        self.priors = priors

    def _initial_fit(self, X, targets):
        """
        """

        # fit the priors
        self._class_prior_fit(X, targets)
 
        self.v_size_ = X.shape[1]

        # Compute y per target value
        self.y_nex_ = np.zeros((n_classes, v_size)) 
        self.log_kmer_probs_ = np.zeros(self.y_nex_.shape)

        for ind in range(n_classes):
            X_class = X[targets == self.classes_[ind]]
            # sum word by word
            self.y_nex_[ind, :] = np.sum(X_class, axis=0)

        # compute the sum of ys
        self.Y_nex_ = self.y_nex_.sum(axis=1)

        # compute y and Y for backoff kmer word
        self.y_prev_ = compute_backoff_words(self.y_nex_)
        self.Y_prev_ = self.y_prev_.sum(axis=1)

        return self
 
    def _log_joint_prob_density(self, X):
        """
        Compute the unnormalized posterior log probability of sequence
 
        I.e. ``log P(C) + log P(sequence | C)`` for all rows x of X, as an array-like of
        shape [n_sequences, n_classes].

        Input is passed to _log_joint_prob_density as-is by predict,
        predict_proba and predict_log_proba. 
        """

        return (np.dot(X, self.log_nex_probs_.T) - np.dot(X, self.log_prev_probs_.T)) + self.log_class_priors_


class MLE_MarkovModel(BaseMarkovModel):
    """
    """
    
    def fit(self, X, targets):
        self._initial_fit(X, targets)

        self.log_nex_probs_ = np.nan_to_num(np.log(self.y_nex_))
        self.log_prev_probs_ = np.nan_to_num(np.log(self.y_prev_)) 

        return self


class Bayesian_MarkovModel(BaseMarkovModel):
    """
    """

    def __init__(self, priors=None, alpha=1e-10):
        super().__init__(priors=priors)
        # validate alpha
        self.alpha = check_alpha(alpha)

    def fit(self, X, targets):
        self._initial_fit(X, targets)

        self.log_nex_probs_ = np.log(self.y_nex_ + self.alpha) 
        self.log_prev_probs_ = np.log(self.y_prev_ + self.alpha) 

        return self

 
# this function will be removed and replaced by a subclass
# BaseMarkovModel
def markov_chain_estimation(sequences, all_kmers, k):
    from kmers import FullKmersCollection
    from utils import get_index_from_kmer
    import numpy as np
    
    v_size = len(all_kmers)

    # targets
    targets = np.asarray(sequences.targets)
    u_classes = np.unique(targets)
    n_classes = len(u_classes)
 
    prior_alpha = np.zeros((n_classes, v_size))

    # Compute kmer word counts for k and k-1
    # n order
    words_k_0 = FullKmersCollection(sequences, k=k)
    clf_k_0 = MLE_MultinomialNaiveBayes(priors="ones").fit(words_k_0.data, targets)
    all_kmers_c = FullKmersCollection(all_kmers, k=k).data
    log_probs_c = clf_k_0._log_joint_prob_density(all_kmers_c)

    # n-1 order
    words_k_1 = FullKmersCollection(sequences, k=(k - 1))
    clf_k_1 = MLE_MultinomialNaiveBayes(priors="ones").fit(words_k_1.data, targets)
    all_kmers_c_1 = FullKmersCollection(all_kmers, k=(k - 1)).data
    log_probs_c_1 = clf_k_1._log_joint_prob_density(all_kmers_c_1)


    final_logs = log_probs_c - log_probs_c_1
    final_prob = np.exp(final_logs)

    #for i, kmer in enumerate(all_kmers):
    #    kmer_ind = get_index_from_kmer(kmer, main_k)
    #    
    #    prior_alpha[kmer_ind] = np.exp(final_logs[i])

    # normalize
    return final_prob/final_prob.sum(axis=0, keepdims=True)
    #return prior_alpha
