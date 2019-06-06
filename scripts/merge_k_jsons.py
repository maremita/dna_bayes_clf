#!/usr/bin/env python

from os import listdir
from os.path import isfile, join
import sys
import json
from collections import defaultdict


if __name__ == "__main__":
    """
    merge_k_jsons.py 
    ~/Projects/Thesis/dna_bayes_clf/results/viruses/wcb_2019/graham/HIV01_FT_250 
    HIV01_FT_250 
    ~/Projects/Thesis/dna_bayes_clf/results/viruses/wcb_2019/graham/HIV01_FT_250/HIV01_FT_250.json
    """

    k_scores = defaultdict(dict)
    final_scores = defaultdict(dict)

    my_path = sys.argv[1]
    prefix = sys.argv[2]
    output_file = sys.argv[3]

    # Get files
    files= [f for f in listdir(my_path) if isfile(join(my_path, f)) and 
            f.endswith(".json") and f.startswith(prefix)]

    for one_file in files:
        scores = json.load(open(join(my_path, one_file), "r"))

        for clf_dscp in scores:
            for str_k in scores[clf_dscp]:
                k_scores[clf_dscp][str_k] = scores[clf_dscp][str_k]
    
    # Sort Ks
    for clf_dscp in k_scores:
        for str_k in sorted(k_scores[clf_dscp], key=lambda v: int(v)):
            final_scores[clf_dscp][str_k] = k_scores[clf_dscp][str_k]

    with open(output_file ,"w") as fh_out: 
        json.dump(final_scores, fh_out, indent=2)
