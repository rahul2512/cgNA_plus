import sys, random, re
#print(sys.path)
sys.path.append('./../classes')
sys.path.append('./../modules')
from cgDNAclass import cgDNA


seq1 = "GCATCATCAGACTACTACTCAGC"  # randomly chosen sequence

data1 = cgDNA(seq1,"dna_ps2")

