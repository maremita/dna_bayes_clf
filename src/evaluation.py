
from src import seq_collection
from src import kmers
from src import utils

from sklearn.model_selection import StratifiedShuffleSplit

def seq_dataset_construction(seq_file, cls_file, k_main, k_estim,
        random_state=None, verbose=False):

    data_seqs = seq_collection.SeqClassCollection((seq_file, cls_file))

    # Split sequences for estimation and cv steps
    seq_ind = list(i for i in range(0,len(data_seqs)))
    a, b = next(StratifiedShuffleSplit(n_splits=1, test_size=0.1,
        random_state=random_state).split(seq_ind, data_seqs.targets))

    seq_cv = data_seqs[list(a)]
    seq_estim = data_seqs[list(b)]

    # Construct the dataset for alpha  estimation
    seq_estim_data = kmers.FullKmersCollection(seq_estim, k=k_estim).data
    seq_estim_targets = seq_estim.targets

    # Construct the data for cross-validation
    seq_cv_data = kmers.FullKmersCollection(seq_cv, k=k_main).data
    seq_cv_targets = seq_cv.targets

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

