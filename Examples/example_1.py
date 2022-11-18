import sys, random, re
#print(sys.path)
sys.path.append('./../classes')
sys.path.append('./../modules')
from cgDNAclass import cgDNA
from cgDNAUtils import *


seq1 = "GCATCATCAGACTACTACTCAGC"  # randomly chosen sequence
data1 = cgDNA(seq1,"dna_ps2")
data2 = cgDNA(seq1,"rna_ps2")  #internally converts T to U
data3 = cgDNA(seq1,"drh_ps2")  #considering DNA strand as reading strand

data1.plot2D()
data1.plot3D()

NbrSamples=1000  ##number of monte carlo samples
Drop_base=0      ## how many bases to ignore from the end
data1.MonteCarlo(NbrSamples,Drop_base) #monte carlo to compute tangent-tangent correlation and persistence length (for heavier computation use cgNA+mc)
data1.makePDB()

## computing groovewidths
gs = GrooveWidths_CS(data1.ground_state)
print("Groovewidth (minor and major) for ",seq1," is ", gs)
