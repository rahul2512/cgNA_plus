README
-------------------------------------------------------
 cgNA+ python (2022) --- developed at LCVMM, EPFL (https://lcvmwww.epfl.ch/)
-------------------------------------------------------    
 Public version: last updated by Rahul Sharma (rs25.iitr@gmail.com, rahul.sharma@epfl.ch) on Nov 2022
-------------------------------------------------------


cgNA+ is a software (Matlab or Octave or Python) package for
predicting the ground-state conformation and stiffness
matrix for double-stranded nucleic acids (dsNAs) fragment of any given sequence.
Free energy differences between two configurations of a
molecule of dsNA in standard environmental conditions, can then be computed.

The ground-state conformation is provided in the CURVES+
definition (for the bases and similar coordinates for phosphates) 
of the dsNA structural coordinates (both a
non-dimensional version and the original unscaled version),
and also as a PDB file of atomic coordinates. The PDB file
can be used in the program 3DNA to obtain 3DNA structural
coordinates if desired. The ground-state stiffness matrix
is provided for non-dimensional version of the CURVES+ helical coordinates.

#----------------------------------------------------------------------------
A user-friendly web version of this program is available at cgDNAweb.epfl.ch

If you use cgDNAweb.epfl.ch in relation to any publication please cite:

cgNA+web: A web based visual interface to the cgNA+ sequence dependent statistical mechanics model of double-stranded nucleic acids
R. Sharma, A. S. Patelli, L. De Bruin, and J.H. Maddocks
Submitted
The cgNA+web site is an evolution from a pre-cursor site (still available at https://cgdnaweb.epfl.ch/view2) and is described in:

cgDNAweb: a web interface to the cgDNA sequence-dependent coarse-grain model of double-stranded DNA.
L. De Bruin, J.H. Maddocks
Nucleic Acids Research 46, issue W1 (2018), p. W5-W10
DOI:10.1093/nar/gky351
#----------------------------------------------------------------------------

The current updated cgNA+web version is an interface to the enhanced coarse-grain model cgNA+ of sequence-dependent statistical mechanics of double-stranded nucleic acids. cgNA+ includes parameter sets for dsDNA in an epigenetic sequence alphabet, dsRNA, and DNA:RNA hybrid as described in detail in

cgNA+: A sequence-dependent coarse-grain model of double-stranded nucleic acids.
R. Sharma, EPFL Thesis #9792, Under the supervision of J. H. Maddocks
Download the PDF here https://lcvmwww.epfl.ch/publications/data/phd/18/PhD_thesis_final.pdf
The extended cgNA+ parameter sets are built on the cgDNA+ model, which itself extends the original cgDNA model by the inclusion of an explicit description of phosphate groups. The cite for the cgDNA+ model itself is:

A sequence-dependent coarse-grain model of B-DNA with explicit description of bases and phosphate groups parametrised from large scale Molecular Dynamics simulations.
A. S. Patelli, EPFL Thesis #9522, Under the supervision of J. H. Maddocks
Download the PDF here https://lcvmwww.epfl.ch/publications/data/phd/15/EPFL_TH9552.pdf
More generally the cgDNA family of models has its own web page, which includes citations to all other related codes and articles.

More information is available at

http://lcvmwww.epfl.ch/cgDNA
#----------------------------------------------------------------------------


%-------------------------------------------------------
% For the impatient...
%-------------------------------------------------------
Run the examples_i.py in Examples directory and also see the basic description of the functions in input.py. 
For more details, please read the codes in functions directory. 

%-------------------------------------------------------
% cgDNA package contents
%-------------------------------------------------------
Four basic directories:
1. Functions: contains all the necessary functions
2. Parametersets: Conatains the different parameter sets obtaind from different MD simulations. 
3. Examples: contains all the input files and output file for few example sequences. 
4. work: This is the working directory. You can change its name and just change the necessary 
	 updates in the input.py file. 

Then there are two text files: 1. README(this file) 2. INSTALLATION file 

1. Functions: 

a) constructSeqParms.py : This function predicts the groundstate and stiffness matrix of the input sequence. 
			  To obtain the results in .mat format; uncomment last line of this script.
			  
b) vector2shapes.py     : This function re-orders the ground-state coordinates.
c) frames.py		: Given a ground-state coordinate vector in non-dimensional Curves+ form, this 
                          function constructs a reference point and frame for each base on each strand of 
		          the DNA according to the Tsukuba convention.    
d) seq_edit.py  	: Given the compacted input sequence, this function expands it. 
e) wcc.py 		: Given the input watson strand, it gives Crick strand or vice-versa. 
f) cgDNA2dplot.py       : Given the input data for the sequence(s), it plots the ground-state 
			  coordinates to the screen for all the input sequences. 
			  To create publication quality plots: inter.png and intra.png in 1200dpi.
				uncomment last two line in this script.  
g) cgDNA3dplot.py	: This function plots the 3D reconstruction of the ground states as rigid
 			  bodies colored according to the sequence and the Crick Watson pairing
			  To create a 1200 dpi png with the name "3D.png".
h) cgDNA_MonteCarlo.py  : This function draws a given number of configurations from the cgDNA probability 
			  density function for the sequence(s) in the given Data structure.
			  This also calculates apparent and dynamic persistent lengths for the given input
			  sequence. One can also obtain the tangent-tangent correlation data. 
i) makepdb.py           : This function constructs the ideal coordinates of the non-hydrogen atoms of each 
			  base according to the Tsukuba definition, and writes the output to a PDB file 
			  (backbone atoms are not included).  
j) shapes2vec.py        : This function re-orders any configuration coordinates.
k) nondim2cur.py        : This function transforms the ground-state coordinates from non-dimensional 
			  Curves+ form to the standard (dimensional) Curves+ form.

l) ideal_bases.txt      : The text file with the ideal coordinates (in base frame) of the non-hydrogen 
			  atoms of the bases T, A, C, G, U, M, N, H, K.

2. Parametersets	: Several parameter sets derived from different MD simulations sets and for variety of nucleic acids (see more details below)

3. work   		: you can rename it to any directory and work but you must have input.py in this 
			  directory to run cgDNA and the keyword "current_dir" should be assigned the name
			  of this woring directory. 
4. Examples		: This file contains two sample sequences and all the outputs. One can use this 
			  example as a template for further use. 


This python package is originally developed by Rahul Sharma (rs25.iitr@gmail.com) with help from Alessandro patelli. 


#---------------------------------------
#Details on parameter sets
#---------------------------------------

[dna_ps1] DNA PS1 (cgDNA+ model)
- Palindromic sequence library
- 3 microseconds of Amber MD time series
- bsc1 force field and SPC/E water model
- 150mM of K+ counter-ions (Dang parameters)
- Maximum entropy/likelihood truncation
- Fitting functional: Kullback-Leibler divergence with model pdf in first argument.
- Dinucleotide model with specific blocks for the dimers at the ends.

[dna_ps2] DNA PS2 (cgNA+ model, recommended)
- Palindromic sequence library
- 10 microseconds of Amber MD time series
- bsc1 force field, TIP3P water model
- 150mM of K+ counter-ions (Joung and Cheatham parameters)
- Maximum entropy/likelihood truncation
- Fitting functional: Kullback-Leibler divergence with model pdf in first argument.
- Dinucleotide model with specific blocks for the dimers at the ends.
- Also, contains parameters for modified CpG steps
- C is referred to as M and H when methylated and hydroxymethylated, respectively and
- G is referred to as N and K when complementary C is methylated and hydroxymethylated, respectively
- Note only CpG steps can be modified i.e. allowed steps are MN, MG, CN or HK, HG, CK 
- hydroxymethylated and methylated steps are allowed in the same sequence but not adjacent

[rna_ps2] RNA PS2 (cgNA+ model, recommended)
- Palindromic sequence library
- 10 microseconds of Amber MD time series
- OL3 force field, TIP3P water model
- 150mM of K+ counter-ions (Joung and Cheatham parameters)
- Maximum entropy/likelihood truncation
- Fitting functional: Kullback-Leibler divergence with model pdf in first argument.
- Dinucleotide model with specific blocks for the dimers at the ends.
- input accepts both U and T but then internally change T to U

[drh_ps2] DNA:RNA Hybrid (DRH) PS2 (cgNA+ model, recommended)
- Same sequence library but not palindromic
- 10 microseconds of Amber MD time series
- bsc1 and OL3 force field for DNA and RNA strand, respectively, TIP3P water model
- 150mM of K+ counter-ions (Joung and Cheatham parameters)
- Maximum entropy/likelihood truncation
- Fitting functional: Kullback-Leibler divergence with model pdf in first argument.
- Dinucleotide model with specific blocks for the dimers at the ends (only GC ends)
- Only accepts sequence in A, T, C, G and must be with GC ends

