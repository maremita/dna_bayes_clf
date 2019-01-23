import seq_collection
import kmers

seq = "ACGTACGTACGTACGTACGTA"
print(kmers.Kmers(seq, n=6, unit_size=1))


cls_file = "data/viruses/HBV/HBV_geo.csv"
seq_file = "data/viruses/HBV/HBV_geo.fasta"

seqco = seq_collection.SeqClassCollection((seq_file, cls_file))

k6mer = kmers.KmersCollection(seqco, n=6, unit_size=1, alphabet="ACGT")
k2mer = kmers.KmersCollection(seqco, n=2, unit_size=1, alphabet="ACGT")

