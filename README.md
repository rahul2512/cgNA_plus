# cgNA+ Python (2022)

Developed at [LCVMM, EPFL](https://lcvmwww.epfl.ch/).

The cgNA+ model is an evolution of the cgDNA family of models developed in the LCVMM laboratory of Prof. John H. Maddocks at EPFL.

This Python package was originally developed by Rahul Sharma (rs25.iitr@gmail.com), with contributions from Alessandro Patelli.

**Public version:** Last updated by Rahul Sharma (rs25.iitr@gmail.com, rahul.sharma@epfl.ch) in November 2022.

The MATLAB version is available at [cgNA_plus_Matlab](https://github.com/rahul2512/cgNA_plus_Matlab).

## Overview

cgNA+ is a software package available in MATLAB, Octave, and Python for predicting the ground-state conformation and stiffness matrix of a double-stranded nucleic acid (dsNA) fragment with any given sequence. Free-energy differences between two configurations of a dsNA molecule under standard environmental conditions can then be computed.

The ground-state conformation is provided in CURVES+ structural coordinates for the bases, with analogous coordinates for the phosphates, in both non-dimensional and original unscaled forms. It is also provided as a PDB file containing atomic coordinates. The PDB file can be used with 3DNA to obtain 3DNA structural coordinates, if desired. The ground-state stiffness matrix is provided in the non-dimensional form of the CURVES+ helical coordinates.

## Web Interface and References

A user-friendly web version of the program is available at [cgDNAweb.epfl.ch](https://cgdnaweb.epfl.ch/).

If you use cgDNAweb.epfl.ch in a publication, please cite:

**cgNA+web: A web based visual interface to the cgNA+ sequence dependent statistical mechanics model of double-stranded nucleic acids**  
R. Sharma, A. S. Patelli, L. De Bruin, and J. H. Maddocks  
*Submitted*

The cgNA+web site evolved from a precursor site, still available at [cgdnaweb.epfl.ch/view2](https://cgdnaweb.epfl.ch/view2), which is described in:

**cgDNAweb: a web interface to the cgDNA sequence-dependent coarse-grain model of double-stranded DNA.**  
L. De Bruin and J. H. Maddocks  
*Nucleic Acids Research*, 46, issue W1 (2018), pp. W5–W10  
DOI: [10.1093/nar/gky351](https://doi.org/10.1093/nar/gky351)

The updated cgNA+web version provides an interface to the enhanced cgNA+ coarse-grain model of sequence-dependent statistical mechanics of double-stranded nucleic acids. cgNA+ includes parameter sets for dsDNA with an epigenetic sequence alphabet, dsRNA, and DNA:RNA hybrids, as described in:

**cgNA+: A sequence-dependent coarse-grain model of double-stranded nucleic acids.**  
R. Sharma, EPFL Thesis #9792, under the supervision of J. H. Maddocks  
[Download PDF](https://lcvmwww.epfl.ch/publications/data/phd/18/PhD_thesis_final.pdf)

The extended cgNA+ parameter sets are built on the cgDNA+ model, which extends the original cgDNA model by explicitly describing phosphate groups. The cgDNA+ model is described in:

**A sequence-dependent coarse-grain model of B-DNA with explicit description of bases and phosphate groups parametrised from large scale Molecular Dynamics simulations.**  
A. S. Patelli, EPFL Thesis #9522, under the supervision of J. H. Maddocks  
[Download PDF](https://lcvmwww.epfl.ch/publications/data/phd/15/EPFL_TH9552.pdf)

The cgDNA family of models also has a [dedicated web page](http://lcvmwww.epfl.ch/cgDNA) with citations to related codes and articles.

## Installation

The `cgNA_plus` package can be installed locally from the GitHub repository using the following steps.

### 1. Clone the repository

```bash
gh repo clone rahul2512/cgNA_plus
cd cgNA_plus
```

### 2. Install the package

```bash
python -m pip install -e .
```

This installs `cgNA_plus` and its required dependencies in editable mode. Changes to the local source code are automatically reflected without requiring reinstallation.

### 3. Import the package

Once installed, `cgNA_plus` can be imported from any working directory within the same Python environment.

```python
from cgNA_plus.classes.cgDNAclass import cgDNA
import cgNA_plus.modules.cgDNAUtils as tools
```

**Note:** The `-e` option enables editable installation and is recommended for development. For a standard installation, omit `-e`.

## Quick Start

Run `examples_i.py` in the `Examples` directory, and see `input.py` for basic descriptions of the functions.

For more details, refer to the code in the `Functions` directory.

## Package Contents

The package has four main directories:

1. **Functions:** Contains all the necessary functions.
2. **Parametersets:** Contains the parameter sets obtained from different MD simulations.
3. **Examples:** Contains input and output files for a few example sequences.
4. **work:** The working directory. You can rename it by making the necessary updates in `input.py`.

There are also two text files: `README` (this file) and `INSTALLATION`.

### 1. Functions

- **`constructSeqParms.py`:** Predicts the ground state and stiffness matrix for the input sequence. To obtain results in `.mat` format, uncomment the last line of the script.
- **`vector2shapes.py`:** Reorders the ground-state coordinates.
- **`frames.py`:** Given a ground-state coordinate vector in non-dimensional CURVES+ form, constructs a reference point and frame for each base on each DNA strand according to the Tsukuba convention.
- **`seq_edit.py`:** Expands a compacted input sequence.
- **`wcc.py`:** Given the input Watson strand, returns the Crick strand, or vice versa.
- **`cgDNA2dplot.py`:** Plots the ground-state coordinates for all input sequences on the screen. To create publication-quality plots (`inter.png` and `intra.png`) at 1200 dpi, uncomment the last two lines of the script.
- **`cgDNA3dplot.py`:** Plots a 3D reconstruction of the ground states as rigid bodies, coloured according to the sequence and Crick–Watson pairing. A 1200 dpi PNG can be created with the name `3D.png`.
- **`cgDNA_MonteCarlo.py`:** Draws a specified number of configurations from the cgDNA probability density function for the sequence(s) in the given Data structure. It also calculates apparent and dynamic persistence lengths and can provide tangent–tangent correlation data.
- **`makepdb.py`:** Constructs ideal coordinates for the non-hydrogen atoms of each base according to the Tsukuba definition and writes them to a PDB file. Backbone atoms are not included.
- **`shapes2vec.py`:** Reorders the coordinates of any configuration.
- **`nondim2cur.py`:** Transforms ground-state coordinates from non-dimensional CURVES+ form to standard (dimensional) CURVES+ form.
- **`ideal_bases.txt`:** Contains ideal coordinates, in the base frame, for the non-hydrogen atoms of T, A, C, G, U, M, N, H, and K.

### 2. Parametersets

Contains several parameter sets derived from different MD simulations for a variety of nucleic acids. See [Details on Parameter Sets](#details-on-parameter-sets) below.

### 3. work

You can rename this directory. To run cgDNA, `input.py` must be present in the working directory, and the `current_dir` keyword must be set to the name of that directory.

### 4. Examples

Contains two sample sequences and all their outputs. These examples can be used as templates for further work.

## Details on Parameter Sets

### `dna_ps1` — DNA PS1 (cgDNA+ model)

- Palindromic sequence library.
- 3 microseconds of Amber MD time series.
- bsc1 force field and SPC/E water model.
- 150 mM K+ counter-ions (Dang parameters).
- Maximum entropy/likelihood truncation.
- Fitting functional: Kullback–Leibler divergence with the model PDF in the first argument.
- Dinucleotide model with specific blocks for the dimers at the ends.

### `dna_ps2` — DNA PS2 (cgNA+ model, recommended)

- Palindromic sequence library.
- 10 microseconds of Amber MD time series.
- bsc1 force field and TIP3P water model.
- 150 mM K+ counter-ions (Joung and Cheatham parameters).
- Maximum entropy/likelihood truncation.
- Fitting functional: Kullback–Leibler divergence with the model PDF in the first argument.
- Dinucleotide model with specific blocks for the dimers at the ends.
- Also contains parameters for modified CpG steps:
  - C is represented by M when methylated and H when hydroxymethylated.
  - G is represented by N and K when the complementary C is methylated and hydroxymethylated, respectively.
  - Only CpG steps can be modified; the allowed steps are MN, MG, CN or HK, HG, CK.
  - Hydroxymethylated and methylated steps can occur in the same sequence, but not adjacent to each other.

### `rna_ps2` — RNA PS2 (cgNA+ model, recommended)

- Palindromic sequence library.
- 10 microseconds of Amber MD time series.
- OL3 force field and TIP3P water model.
- 150 mM K+ counter-ions (Joung and Cheatham parameters).
- Maximum entropy/likelihood truncation.
- Fitting functional: Kullback–Leibler divergence with the model PDF in the first argument.
- Dinucleotide model with specific blocks for the dimers at the ends.
- Input accepts both U and T, but internally converts T to U.

### `drh_ps2` — DNA:RNA Hybrid (DRH) PS2 (cgNA+ model, recommended)

- Same sequence library, but not palindromic.
- 10 microseconds of Amber MD time series.
- bsc1 and OL3 force fields for the DNA and RNA strands, respectively, with the TIP3P water model.
- 150 mM K+ counter-ions (Joung and Cheatham parameters).
- Maximum entropy/likelihood truncation.
- Fitting functional: Kullback–Leibler divergence with the model PDF in the first argument.
- Dinucleotide model with specific blocks for the dimers at the ends (only GC ends).
- Accepts only sequences with A, T, C, and G, and requires GC ends.
