from os.path import splitext
import re
from collections import UserList, defaultdict

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

__all__ = ['SeqClassCollection']


class SeqClassCollection(UserList):
    
    def __init__(self, arg):

        self.data = []
        self.targets = []
        self.target_map = {}

        # If arguments are two files
        # Fasta file and annotation file
        if isinstance(arg, tuple):
            self.data = self.read_bio_file(arg[0])
            self.target_map = self.read_class_file(arg[1])
            self.set_targets()

        # If argument is a list of labeled seq records 
        elif isinstance(arg, list):
            self.data = arg
            self.get_targets()

        # If argument is SeqClassCollection object
        elif isinstance(arg, self.__class__):
            self.data = arg.data[:]
            self.get_targets()

        else:
            self.data = list(arg)
            self.get_targets() 

    def set_targets(self):
        for seqRecord in self.data:
            if seqRecord.id in self.target_map:
                seqRecord.target = self.target_map[seqRecord.id]
                self.targets.append(self.target_map[seqRecord.id])

            else:
                print("No target label for {}\n".format(seqRecord.id))
                self.targets.append("UNKNOWN")

    def get_targets(self):

        self.target_map = dict((seqRec.id, seqRec.target)
                        for seqRec in self.data)

        self.targets = list(seqRec.target for seqRec in self.data)

    def __getitem__(self, ind):
        # TODO
        # Give more details about this exception
        if not isinstance(ind, (int, list, slice)):
            raise TypeError("The argument must be int, list or slice")

        # shallow copy 
        #if the argument is an integer
        if isinstance(ind, int):
            return self.data[ind]

        # If the argument is a list of indexes
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

    def get_fragments(self, size, step=1):

        if step < 1:
            print("do_fragment step parameter should be sup to 1")
            step = 1

        new_data = []

        for ind, seqRec in enumerate(self.data):
            sequence = seqRec.seq
 
            i = 0
            while i < (len(sequence) - size + 1):
                fragment = sequence[i:i + size]

                frgRec = SeqRecord(fragment, id=seqRec.id+"_"+str(i))
                frgRec.rankParent = ind
                frgRec.seqParent = seqRec.id
                frgRec.target = seqRec.target
                new_data.append(frgRec)
                i += step

        return self.__class__(new_data)

    def get_parents_rank_list(self):
        parents = defaultdict(list)

        for ind, seqRec in enumerate(self.data):
            if hasattr(seqRec, "rankParent"):
                parents[seqRec.rankParent].append(ind)

        return parents


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

