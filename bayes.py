import kmers
import numpy as np


class BaseNaiveBayes():
    """
    """

    def __init__(self):
        pass

    def _initial_fit(self, sequences, main_k, secd_k):
        """
        """

        # ordered list of target labels
        targets = np.asarray(sequences.targets)
        self.classes_ = np.unique(targets)
        n_classes = len(self.classes_)

        # compute kmers
        main_kmer = kmers.KmersCollection(sequences, n=main_k, alphabet="ACGT")
        secd_kmer = kmers.KmersCollection(sequences, n=main_k, alphabet="ACGT")

        # compute y per target value
        self.y = np.zeros((n_classes, len(main_kmer.kmers_list))) 

        for ind in range(n_classes):
            X_class = main_kmer.data[targets == self.classes_[ind]]
            # sum word by word
            self.y[ind, :] = np.sum(X_class, axis=0)
        
        # compute the sum of ys
        self.Y = np.zeros(n_classes)

        for ind in range(n_classes):
            self.Y[ind] = np.sum(y[ind, axis=0])

        return self

#    def _log_joint_prob_density(self, X):
#        pass
#
#    def predict(self, X):
#        """
#        Perform classification on an array of test vectors X.
#
#        Parameters
#        ----------
#        X : array-like, shape = (n_samples, n_features)
#
#        Returns
#        -------
#        C : array, shape = (n_samples)
#            Predicted target values for X
#        """
#
#        ljb = self._log_joint_prob_density(X)
#        # ljb has a shape of (n_samples, n_classes)
#
#        return self.classes_[np.argmax(ljb, axis=1)]
#
#    def predict_log_proba(self):
#        """
#        Return log-probability estimates for the test vector X.
#        
#        Parameters
#        ----------
#        X : array-like, shape = (n_samples, n_features)
#        
#        Returns
#        -------
#        C : array-like, shape = (n_samples, n_classes)
#            Returns the log-probability of the samples for each class in
#            the model. The columns correspond to the classes in sorted
#            order, as they appear in the attribute `classes_`.
#        """
#
#        ljb = self._log_joint_prob_density(X)
#
#        # normalize by P(x) = P(f_1, ..., f_n) la formule est fausse
#        # We use marginalization to compute P(x)
#        # P(x) = sum(P(x, y_i))
#
#        # ljb contains log joint prob densitys for each class
#        # We put this values in exp and sum them
#        # finally, we compute the log of the sum
#        # logsumexp : Compute the log of the sum of exponentials of 
#        #             input elements.
#        #
#        # we substract log_prob_x because we calculate with logs, if we work
#        # with densities or probabilities we divide by P(x)
#
#        log_prob_x = logsumexp(ljb, axis=1)
#
#        return ljb - np.atleast_2d(log_prob_x).T
#
#    def predict_proba(self):
#        """
#        Return probability estimates for the test vector X.
# 
#        Parameters
#        ----------
#        X : array-like, shape = (n_samples, n_features)
#
#        Returns
#        -------
#        C : array-like, shape = (n_samples, n_classes)
#            Returns the probability of the samples for each class in
#            the model. The columns correspond to the classes in sorted
#            order, as they appear in the attribute `classes_`.
#        """
#
#        return np.exp(self.predict_log_proba(X))

class MLE_MultinomialNaiveBayes(BaseNaiveBayes):
    """
    """
    
    def fit(self, sequences, main_k, secd_k):
        self._initial_fit(self, sequences, main_k, secd_k)
        

        return self




    


