import sys, random, re
#print(sys.path)
sys.path.append('./../classes')
sys.path.append('./../modules')
from cgDNAclass import cgDNA
from cgDNAUtils import *


seq1 = "GCATCATCAGACTACTACTCAGC"  # randomly chosen sequence
data1 = cgDNA(seq1,"dna_ps2")
data2 = cgDNA(seq1,"rna_ps2")
data3 = cgDNA(seq1,"drh_ps2")

gs = GrooveWidths_CS(data1.ground_state)

print("Groovewidth (minor and major) for ",seq1," is ", gs)
