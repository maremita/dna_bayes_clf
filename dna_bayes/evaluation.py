from dna_bayes import seq_collection
from dna_bayes import kmers
from dna_bayes import utils

import os.path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.model_selection import StratifiedShuffleSplit


def construct_split_collection(seq_file, cls_file, estim_size, random_state=None):

    data_seqs = seq_collection.SeqClassCollection((seq_file, cls_file))

    # Split sequences for estimation and cv steps
    seq_ind = list(i for i in range(0,len(data_seqs)))
    a, b = next(StratifiedShuffleSplit(n_splits=1, test_size=estim_size,
        random_state=random_state).split(seq_ind, data_seqs.targets))

    seq_cv = data_seqs[list(a)]
    seq_estim = data_seqs[list(b)]

    return seq_cv, seq_estim


def seq_dataset_construction(seq_file, cls_file, estim_size, k_main, k_estim,
        random_state=None, verbose=False):

    seq_cv, seq_estim = construct_split_collection(seq_file, cls_file, estim_size, random_state=random_state)

    # Construct the data for cross-validation
    seq_cv_data = kmers.FullKmersCollection(seq_cv, k=k_main).data
    seq_cv_targets = seq_cv.targets

    # Construct the dataset for alpha  estimation
    seq_estim_data = kmers.FullKmersCollection(seq_estim, k=k_estim).data
    seq_estim_targets = seq_estim.targets

    return seq_cv_data, seq_cv_targets, seq_estim_data, seq_estim_targets


def kmer_dataset_construction(k_main, k_estim,
        alphabet='ACGT', verbose=False):

    # Get kmer word list
    all_words = utils.generate_all_words(alphabet, k_main)
    all_words_data = kmers.FullKmersCollection(all_words, k=k_estim).data

    # Get kmer word for backoff
    all_backs = utils.generate_all_words(alphabet, k_main-1)
    all_backs_data = kmers.FullKmersCollection(all_backs, k=k_estim).data

    return all_words_data, all_backs_data


def make_figure(scores, clfNames, kList, jsonFile, verbose=True):
    if verbose: print("generating a figure")
    
    fig_file = os.path.splitext(jsonFile)[0] + ".png"
    fig_title = os.path.splitext((os.path.basename(jsonFile)))[0]
    
    cmap = cm.get_cmap('tab20')
    colors = [cmap(j/10) for j in range(0,10)] 

    f, axs = plt.subplots(2, 5, figsize=(22,10))
    axs = np.concatenate(axs)
    #axs = list(zip(axs,clf_symbs))
    width = 0.45

    for ind, algo in enumerate(scores):
        means = np.array([scores[algo][str(k)][0] for k in kList])
        stds = np.array([scores[algo][str(k)][1] for k in kList])

        #axs[ind].bar(kList, means, width, yerr=stds)
        
        axs[ind].fill_between(kList, means-stds, means+stds, alpha=0.1, color=colors[ind])
        axs[ind].plot(kList, means, color=colors[ind])

        axs[ind].set_title(clfNames[algo])
        axs[ind].set_ylim([0,1.1])
        axs[ind].grid()
        
        if ind == 0 or ind == 5:
            axs[ind].set_ylabel('F1 weighted')
        if ind >= 5:
            axs[ind].set_xlabel('K length')
 
    plt.suptitle(fig_title)
    plt.savefig(fig_file)
    plt.show()

