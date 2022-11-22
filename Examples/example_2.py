import sys, random, re
#print(sys.path)
sys.path.append('./../classes')
sys.path.append('./../modules')
from cgDNAclass import cgDNA
from cgDNAUtils import *


seq1 = "GCATCGCGCGCGCGCGCGCGAGC"  # random artificial CpG islands sequence
seq2 = "GCATCGMNMNMNMNMNMNMNAGC" ## symmetric methylation of CpG steps
seq3 = "GCATCGHKHKHKHKHKHKHKAGC" ## symmetric hydroxymethylation of CpG steps
seq4 = "GCATCGMNCGCNCGCGMGCGAGC" ## Asymmetric methylation of CpG steps 
## see README file for precise rules for what kind of steps are allowed

## parameters for modified steps are available in dna_ps2 parameter set

data1 = cgDNA(seq1,"dna_ps2")
data2 = cgDNA(seq2,"dna_ps2")  
data3 = cgDNA(seq3,"dna_ps2")  
data4 = cgDNA(seq4,"dna_ps2")  

data1.plot2D()
data1.plot3D()

NbrSamples=1000  ##number of monte carlo samples
Drop_base=0      ## how many bases to ignore from the end
data1.MonteCarlo(NbrSamples,Drop_base) #monte carlo to compute tangent-tangent correlation and persistence length (for heavier computation use cgNA+mc)
data1.makePDB()

## computing groovewidths
gs = GrooveWidths_CS(data1.ground_state)
print("Groovewidth (minor and major) for ",seq1," is ", gs)
