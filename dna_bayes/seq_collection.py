from os.path import splitext
import re
import copy
import random
from collections import UserList, defaultdict

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

__all__ = ['SeqClassCollection']



class SeqClassCollection(UserList):

    """
    Attributes
    ----------

    data : list of Bio.SeqRecord
        Collection of sequence records

    targets : list
        Collection of targets of the sequences
        The order of target needs to be the same as
        the sequences in data

    target_map : dict
        mapping of sequences and their targets (classes)

    taget_ind : defaultdict(list)
        Collection of targets and the indices of belonging
        sequences

    """

    def __init__(self, arg):
 
        self.data = []
        self.targets = []
        self.target_map = {}
        self.target_ind = defaultdict(list)

        # If arguments are two files
        # Fasta file and annotation file
        if isinstance(arg, tuple):
            self.data = self.read_bio_file(arg[0])
            self.target_map = self.read_class_file(arg[1])
            self.set_targets()

        # If argument is a list of labeled seq records 
        elif isinstance(arg, list):
            #self.data = arg
            self.data = copy.deepcopy(arg)
            self.get_targets()

        # If argument is SeqClassCollection object
        elif isinstance(arg, self.__class__):
            #self.data = arg.data[:]
            self.data = copy.deepcopy(arg.data)
            self.get_targets()

        # why?
        else:
            self.data = list(copy.deepcopy(arg))
            self.get_targets() 

    def set_targets(self):
        for ind, seqRecord in enumerate(self.data):
            if seqRecord.id in self.target_map:
                seqRecord.target = self.target_map[seqRecord.id]
                self.targets.append(self.target_map[seqRecord.id])
                self.target_ind[seqRecord.target].append(ind)

            else:
                print("No target label for {}\n".format(seqRecord.id))
                self.targets.append("UNKNOWN")
                self.target_ind["UNKNOWN"].append(ind)

    def get_targets(self):

        self.target_map = dict((seqRec.id, seqRec.target)
                        for seqRec in self.data)

        self.targets = list(seqRec.target for seqRec in self.data)

        for ind, seqRecord in enumerate(self.data):
            self.target_ind[seqRecord.target].append(ind)

    def __getitem__(self, ind):
        # TODO
        # Give more details about this exception
        if not isinstance(ind, (int, list, slice)):
            raise TypeError("The argument must be int, list or slice")

        # shallow copy 
        #if the argument is an integer
        if isinstance(ind, int):
            return self.data[ind]

        # With instantiation, data will be deep copied  
        # If the argument is a list of indices
        elif isinstance(ind, list):

            tmp = [self.data[i] for i in ind if i>= 0 and i<len(self.data)]
            return self.__class__(tmp)

        return self.__class__(self.data[ind])

    @classmethod
    def read_bio_file(cls, my_file):
        path, ext = splitext(my_file)
        ext = ext.lstrip(".")

        if ext == "fa" : ext = "fasta"

        return list(seqRec for seqRec in SeqIO.parse(my_file, ext))

    @classmethod
    def read_class_file(cls, my_file):

        with open(my_file, "r") as fh:
            #return dict(map(lambda x: (x[0], x[1]), (line.rstrip("\n").split(sep)
            return dict(map(lambda x: (x[0], x[1]), (re.split(r'[\t,;\s]', line.rstrip("\n"))
                        for line in fh if not line.startswith("#"))))

    @classmethod
    def write_fasta(cls, data, out_fasta):
        SeqIO.write(data, out_fasta, "fasta")

    @classmethod
    def write_classes(cls, classes, file_class):
        with open(file_class, "w") as fh:
            for entry in classes:
                fh.write(entry+","+classes[entry]+"\n")

    def get_fragments(self, size, step=1):

        if step < 1:
            print("get_fragments() step parameter should be sup to 1")
            step = 1

        new_data = []

        for ind, seqRec in enumerate(self.data):
            sequence = seqRec.seq
 
            i = 0
            j = 0
            while i < (len(sequence) - size + 1):
                fragment = sequence[i:i + size]

                frgRec = SeqRecord(fragment, id=seqRec.id + "_" + str(j))
                frgRec.rankParent = ind
                frgRec.idParent = seqRec.id
                frgRec.target = seqRec.target
                frgRec.description = seqRec.description
                frgRec.name = "{}.fragment_at_{}".format(seqRec.name, str(i))
                frgRec.position = i

                new_data.append(frgRec)
                i += step
                j += 1

        return self.__class__(new_data)

    def get_parents_rank_list(self):
        parents = defaultdict(list)

        for ind, seqRec in enumerate(self.data):
            if hasattr(seqRec, "rankParent"):
                parents[seqRec.rankParent].append(ind)

        return parents

    def sample(self, size, seed=None):
        random.seed(seed)

        if size > len(self.data):
            return self

        else:
            return self.__class__(random.sample(self, size))

    def stratified_sample(self, sup_limit=25, inf_limit=5, seed=None):
        random.seed(seed)

        new_data_ind = []

        for target in self.target_ind:
            nb_seqs = len(self.target_ind[target])
            the_limit = sup_limit
    
            if nb_seqs <= the_limit:
                the_limit = nb_seqs
    
            if nb_seqs >= inf_limit:
                new_data_ind.extend(random.sample(self.target_ind[target], the_limit))

        return self[new_data_ind]

    def write(self, fasta_file, class_file):
       self.write_fasta(self.data, fasta_file)
       self.write_classes(self.target_map, class_file)

if __name__ == "__main__":
    cls_file = "../data/viruses/HBV/HBV_geo.csv"
    seq_file = "../data/viruses/HBV/HBV_geo.fasta"

    # clas = SeqClassCollection.read_class_file(cls_file, "\t")
    # seqs = [ seq for seq in SeqClassCollection.read_bio_file(seq_file) if seq.id in clas]

    # print(clas)
    # with open("../data/viruses/HBV/HBV_geo.fasta", "w") as output_handle:
    #    SeqIO.write(seqs, output_handle, "fasta")

    #seqco = SeqClassCollection((seq_file, cls_file))
    #print(seqco.data[0:3])
    #print(type(seqco.data[0:3]))
 
    # seqs = SeqClassCollection(seq for seq in SeqClassCollection.read_bio_file(seq_file))
    # print(seqs)
    # print(seqs.target_map)
    # print(type(seqs))


