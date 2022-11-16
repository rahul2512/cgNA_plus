README
%-------------------------------------------------------
% cgDNA, version 2 in python (2018)
%-------------------------------------------------------    

cgNA+ is a software (Matlab or Octave or Python) package for
predicting the ground-state conformation and stiffness
matrix of a molecule of double-stranded nucleic acids (dsNAs) of any given sequence.
Free energy differences between two configurations of a
molecule of dsNA in standard environmental conditions, can then be computed.

The ground-state conformation is provided in the CURVES+
definition of the dsNA structural coordinates (both a
non-dimensional version and the original unscaled version),
and also as a PDB file of atomic coordinates. The PDB file
can be used in the program 3DNA to obtain 3DNA structural
coordinates if desired. The ground-state stiffness matrix
is provided for non-dimensional version of the Curves+
helical coordinates.

A user-friendly web version of this program is available at

cgDNAweb.epfl.ch


 

More information is available at

http://lcvmwww.epfl.ch/cgDNA

and in:

D. Petkeviciute, M. Pasi, O. Gonzalez and J.H. Maddocks. 
 cgDNA: a software package for the prediction of
 sequence-dependent coarse-grain free energies of B-form
 DNA. Submitted (2014). 

If you find cgDNA useful, please cite the above
publication.


%-------------------------------------------------------
% For the impatient...
%-------------------------------------------------------
Run the input.py in Examples directory and also see the basic description of the functions in input.py. 
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
			  atoms of the four bases T, A, C, G.

2. Parametersets	: Four parametersets derived from different MD simulations sets.

3. work   		: you can rename it to any directory and work but you must have input.py in this 
			  directory to run cgDNA and the keyword "current_dir" should be assigned the name
			  of this woring directory. 
4. Examples		: This file contains two sample sequences and all the outputs. One can use this 
			  example as a template for further use. 

# cgNA_plus
# Creating cgNA object
# data = cgDNA(seq,ps) where seq is in A, T, C, G, M, N, H, K, U and ps is appropriate parameterset for that nucliec acid. 
# For instanse, choose ps = "ps2_cgf" for the latest DNA parameter set
# ps = "ps_rna" and "ps_hyb" for RNA and DNA:RNA hybrid

