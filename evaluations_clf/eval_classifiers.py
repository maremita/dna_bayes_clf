from dna_bayes import bayes
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC

priors="uniform"

eval_clfs = {
        "bayes":{
            0: [bayes.MLE_MultinomialNB(priors=priors), False, "MLE_MultinomNB"],
            1: [bayes.Bayesian_MultinomialNB(priors=priors, alpha=1e-100), False, "BAY_MultinomNB_Alpha_1e-100"],
            2: [bayes.Bayesian_MultinomialNB(priors=priors, alpha=1e-10), False, "BAY_MultinomNB_Alpha_1e-10"],
            3: [bayes.Bayesian_MultinomialNB(priors=priors, alpha=1e-5), False, "BAY_MultinomNB_Alpha_1e-5"],
            4: [bayes.Bayesian_MultinomialNB(priors=priors, alpha=1), False, "BAY_MultinomNB_Alpha_1"]
            },

        "markov":{
            0: [bayes.MLE_MarkovModel(priors=priors), True, "MLE_Markov"],
            1: [bayes.Bayesian_MarkovModel(priors=priors, alpha=1e-100), True, "BAY_Markov_Alpha_1e-100"],
            2: [bayes.Bayesian_MarkovModel(priors=priors, alpha=1e-10), True, "BAY_Markov_Alpha_1e-10"],
            3: [bayes.Bayesian_MarkovModel(priors=priors, alpha=1e-5), True, "BAY_Markov_Alpha_1e-5"],
            4: [bayes.Bayesian_MarkovModel(priors=priors, alpha=1), True, "BAY_Markov_Alpha_1"]
            },

        "lr":{
            0:  [LogisticRegression(multi_class='ovr', solver='liblinear', penalty="l1", max_iter=1500), False, "SK_Ovr_LR_Liblinear_L1"],
            1: [LogisticRegression(multi_class='ovr', solver='liblinear', penalty="l2", max_iter=1500), False, "SK_Ovr_LR_Liblinear_L2"]
            },

        "lsvm":{
            0: [LinearSVC(penalty="l1", loss="squared_hinge", dual=False, max_iter=1500), False, "SK_LinearSVC_SquaredHinge_L1_Primal"],
            1: [LinearSVC(penalty="l2", loss="hinge", dual=True, max_iter=1500), False, "SK_LinearSVC_Hinge_L2_Dual"],
            2: [LinearSVC(penalty="l2", loss="squared_hinge", dual=True, max_iter=1500), False, "SK_LinearSVC_SquaredHinge_L2_Dual"],
            3: [LinearSVC(penalty="l2", loss="squared_hinge", dual=False, max_iter=1500), False, "SK_LinearSVC_SquaredHinge_L2_Primal"]
            },

        "svm":{
            0: [SVC(kernel="linear"), False, "SK_SVC_Linear_Hinge_L2"],
            1: [SVC(kernel="rbf", gamma="auto"), False, "SK_SVC_RBF"],
            2: [SVC(kernel="poly", gamma="auto"), False, "SK_SVC_Poly"],
            3: [SVC(kernel="sigmoid", gamma="auto"), False, "SK_SVC_Sigmoid"]
            }
        }


        #5: [GaussianNB(), False, "SK_Gaussian_NB"],
        ## 12: [LinearSVC(penalty="l1", loss="hinge"), False, "SK_LinearSVC_Hinge_L1"], # NOT supported
        ##8:  [LogisticRegression(multi_class='multinomial', solver='saga', penalty="l1", max_iter=1500), False, "SK_Multi_LR_Saga_L1"],
        ##10: [LogisticRegression(multi_class='multinomial', solver='saga', penalty="l2", max_iter=1500), False, "SK_Multi_LR_Saga_L2"],
