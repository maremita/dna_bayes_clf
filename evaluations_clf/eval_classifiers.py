from dna_bayes import bayes
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC

priors="uniform"

eval_clfs = {
        0: [bayes.MLE_MultinomialNB(priors=priors), False, "MLE_MultinomNB"],
        1: [bayes.Bayesian_MultinomialNB(priors=priors, alpha=1e-100), False, "BAY_MultinomNB_Alpha_1e-100"],
        2: [bayes.Bayesian_MultinomialNB(priors=priors, alpha=1e-10), False, "BAY_MultinomNB_Alpha_1e-10"],
        3: [bayes.Bayesian_MultinomialNB(priors=priors, alpha=1), False, "BAY_MultinomNB_Alpha_1"],

        4: [bayes.MLE_MarkovModel(priors=priors), True, "MLE_Markov"],
        5: [bayes.Bayesian_MarkovModel(priors=priors, alpha=1e-100), True, "BAY_Markov_Alpha_1e-100"],
        6: [bayes.Bayesian_MarkovModel(priors=priors, alpha=1e-10), True, "BAY_Markov_Alpha_1e-10"],
        7: [bayes.Bayesian_MarkovModel(priors=priors, alpha=1), True, "BAY_Markov_Alpha_1"],

        #5: [GaussianNB(), False, "SK_Gaussian_NB"],
        8:  [LogisticRegression(multi_class='multinomial', solver='saga', penalty="l1", max_iter=1500), False, "SK_Multi_LR_Saga_L1"],
        9:  [LogisticRegression(multi_class='ovr', solver='liblinear', penalty="l1", max_iter=1500), False, "SK_Ovr_LR_Liblinear_L1"],
        10: [LogisticRegression(multi_class='multinomial', solver='saga', penalty="l2", max_iter=1500), False, "SK_Multi_LR_Saga_L2"],
        11: [LogisticRegression(multi_class='ovr', solver='liblinear', penalty="l2", max_iter=1500), False, "SK_Ovr_LR_Liblinear_L2"],
 
        ## 12: [LinearSVC(penalty="l1", loss="hinge"), False, "SK_LinearSVC_Hinge_L1"], # NOT supported
        12: [LinearSVC(penalty="l1", loss="squared_hinge", dual=False), False, "SK_LinearSVC_SquaredHinge_L1_Primal"],
        13: [LinearSVC(penalty="l2", loss="hinge", dual=True), False, "SK_LinearSVC_Hinge_L2_Dual"],
        14: [LinearSVC(penalty="l2", loss="squared_hinge", dual=True), False, "SK_LinearSVC_SquaredHinge_L2_Dual"],
        15: [LinearSVC(penalty="l2", loss="squared_hinge", dual=False), False, "SK_LinearSVC_SquaredHinge_L2_Primal"],

        16: [SVC(kernel="linear"), False, "SK_SVC_Linear_Hinge_L2"],
        17: [SVC(kernel="rbf", gamma="auto"), False, "SK_SVC_RBF"],
        18: [SVC(kernel="poly", gamma="auto"), False, "SK_SVC_Poly"],
        19: [SVC(kernel="sigmoid", gamma="auto"), False, "SK_SVC_Sigmoid"],
            }
