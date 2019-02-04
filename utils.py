import numpy as np

__all__ = ['get_index_from_kmer', 'get_kmer_from_index', 'compute_backoff_words']


# TODO write more generic function (alphabet as parameter)
def get_index_from_kmer(kmer, k):
    """
    Function adapted from module enrich.pyx of
    GenomeClassifier package [Sandberg et al. (2001)]
    
    Instead of starting by f=1 and multiplying it by 4, 
    it starts with f= 4**(k-1) and divide it by 4
    in each iteration
    
    The returned index respects the result of itertools.product()
    ["".join(t) for t in itertools.product('ACGT', repeat=k)]
    """

    f= 4 ** (k-1)
    s=0
    alpha_to_code = {'A':0, 'C':1, 'G':2, 'T':3}

    for i in range(0, k):
        alpha_code=alpha_to_code[kmer[i]]
        s = s + alpha_code * f
        f = f // 4

    return s


def get_kmer_from_index(index, k):
    """
    """
    
    code_to_alpha = {0:'A', 1:'C', 2:'G', 3:'T'}
    num_b4 = base10toN(index, 4)
    l = len(num_b4)

    if l != k:
        num_b4 = [0]*(k-l) + num_b4
   
    return "".join([code_to_alpha[i] for i in num_b4])


def base10toN(num, base):
    l = []
    current = num
    while current:
        mod = current % base
        current = current // base
        l.append(mod)
    l.reverse()

    return l

def compute_backoff_words(X):
    old_k = int(math.log(X.shape[1], 4))
    new_k = old_k - 1
    new_v = int(4**new_k)

    new_X = np.zeros((X.shape[0], new_v))
    #
    for old_ind in range(0, X.shape[1]):
        old_kmer = get_kmer_from_index(old_ind, old_k)
        new_ind = get_index_from_kmer(old_kmer[:-1], new_k)

        new_X[:,new_ind] += X[:,old_ind]
    
    return new_X
