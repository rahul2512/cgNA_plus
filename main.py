import sys
sys.path.append('./modules')
sys.path.append('./classes')
from cgDNAclass import cgDNA
import numpy as np
import cgDNAUtils as tools
import MD_analysis as analysis
from E_transform import Etrans
from Init_MD import init_MD_data
import matplotlib.colors as mcolors

c1 = ['red','blue','green']
c2 = ['darkred','navy','olivedrab']
c3 = ['indianred','royalblue','limegreen']
c4 = ['maroon','dodgerblue','olivedrab']

# A4 column width is 88mm (3.45 in). The space between the two columns is 4mm (0.17 in).
#################### Path for 10 mus final files
PDNA_path     = './BDNA/olig_Palin_141_3us_symmetrized_ver2.mat'
DNA_path      = './BDNA/palin.bscl.tip3p.jc.comb.sym.stats.10mus.cgF.25_tot.mat'
RNA_path      = './BDNA/rna.ol3.tip3p.jc.comb.sym.stats.10mus.cgF.24_tot.mat'    
HYB_path      = './BDNA/hyb.dna.rna.bsc1.ol3.tip3p.jc.comb.stats.sym.10mus.cgF.mat'
DNA_217_path  = './BDNA/palin.bscl.tip3p.jc.comb.sym.stats.20mus.cgF.217.mat'
DNA_ends_path = './BDNA/palin.bscl.tip3p.jc.ends.comb.stats.3mus.cgF.mat'
DNA_long_path = './BDNA/palin.bscl.tip3p.jc.comb.sym.stats.10mus.cgF.26_28.mat'
############### Initiate MD data #################################
PDNA      = init_MD_data().load_data(PDNA_path)
DNA       = init_MD_data().load_data(DNA_path)
DNA_217   = init_MD_data().load_data(DNA_217_path)
DNA_ends  = init_MD_data().load_data(DNA_ends_path)
DNA_long  = init_MD_data().load_data(DNA_long_path)
RNA       = init_MD_data().load_data(RNA_path)
HYB       = init_MD_data().load_data(HYB_path)

############### comparison of ground state of MD vs reconstruction #################################
def main1():
    analysis.plot_gs_vs_MD_shape(DNA_path,RNA_path,HYB_path)

############### persistence lengths plots #################################
def main2():
    p1 = '/Users/rsharma/Dropbox/PhD_work/MD_analysis/persis_len/'

#    analysis.read_persis_file_index(p1+'RNA_OL3_CGF',1)
#    data = analysis.plot_persistence_length_tandem_vs_random(['DNA_ABC_cg'])
#    data = analysis.plot_persistence_length_tandem_vs_random(['DNA_ABC_cgplus'])
#    data = analysis.plot_persistence_length_tandem_vs_random(['DNA_BSTJ_CGF'])
#    data = analysis.plot_persistence_length_tandem_vs_random(['RNA_OL3_CGF'])
#    data = analysis.plot_persistence_length_tandem_vs_random(['HYB_CGF'])

#    data = analysis.plot_special_persistence_length(['DNA_BSTJ_CGF','RNA_OL3_CGF','HYB_CGF'])  ### dimers
    data = analysis.plot_persistence_length(['DNA_BSTJ_CGF','RNA_OL3_CGF','HYB_CGF'])
############### palindromic error ##################
def main3():
    path = './BDNA/palin.bscl.tip3p.jc.comb.stats.XXmus.cgF.mat'
    analysis.plot_heatmap_palin_err(path,name='DNA_cgF')
    # path = './BDNA/rna.ol3.tip3p.jc.comb.stats.sym.XXmus.cgF.mat'
    # analysis.plot_heatmap_palin_err(path,name='RNA_cgF')

############### training/test error #################################
def main4():
    analysis.plot_heatmap_training_err(DNA,'ps2_cgf',sym=True)
    analysis.plot_heatmap_training_err(RNA,'ps_rna',sym=True)
    analysis.plot_heatmap_training_err(HYB,'ps_hyb',sym=False)
    analysis.plot_heatmap_training_err(DNA_long.choose_seq([2]),'ps2_cgf',sym=True)

############### set scale for training/test/palindromic error #################################
def main5():
#    analysis.set_scale_for_error(DNA_path,sym=True)
    analysis.set_scale_for_error(RNA_path,sym=True)
#    analysis.set_scale_for_error(HYB_path,sym=False)

############### compare_same_seq_across_data #################################
linestyles= ['-','-', '--', '-.', ':']
def main6():
    linestyles= ['-',':', '--', '-.', ':']
    col = [c1,c3,c4]
    col = [c1,c1,c1]
    
    for sequence_id in range(17):
        analysis.compare_same_seq_across_data(DNA,RNA,HYB,col,sequence_id,'compare_seq_DNA_RNA_HYB')
    
############### compare_same_seq_across_data #################################
linestyles= ['-','-', '--', '-.', ':']
def main7():
    col = [c1,c1,c4,c4,c3,c3]
    seq_id1=np.array([20,21])
    sym1=[False,False]

    analysis.compare_diff_seq_within_data(DNA,seq_id1-1,'ps2_cgf',col,sym1,'DNA_compare_seq')
    analysis.compare_diff_seq_within_data(RNA,seq_id1-1,'ps_rna',col,sym1,'RNA_compare_seq')
#    analysis.compare_diff_seq_within_data(HYB,seq_id-1,'ps_hyb',col,sym1,'HYB_compare_seq')
    sym2=[False,True]
    seq_id2=np.array([18,19])
    analysis.compare_diff_seq_within_data(DNA,seq_id2-1,'ps2_cgf',col,sym2,'DNA_compare_seq')
    analysis.compare_diff_seq_within_data(RNA,seq_id2-1,'ps_rna',col,sym2,'RNA_compare_seq')
#    analysis.compare_diff_seq_within_data(HYB,seq_id-1,'ps_hyb',col,sym1,'HYB_compare_seq')

    sym3=[True,True]
    seq_id3=np.array([22,23])
    analysis.compare_diff_seq_within_data(DNA,seq_id3-1,'ps2_cgf',col,sym3,'DNA_compare_seq')

    sym4=[True]
    seq_id4=np.array([3])
    analysis.compare_diff_seq_within_data(DNA_long,seq_id4-1,'ps2_cgf',col,sym4,'DNA_long_compare_seq')

##################################---------------------------------------------
# plot oligomer level eigenvalues for DNA,RNA, Hybrid
##################################---------------------------------------------

def main8():
    #### make sure which are sym
    sym=[True,True,False]
    for seq_id in range(17):
        analysis.plot_eig_olig_first(DNA.choose_seq([seq_id]), RNA.choose_seq([seq_id]), HYB.choose_seq([seq_id]), sym, c1, "olig_eig_"+str(1+seq_id))


##################################---------------------------------------------
# plot oligomer level stiffness for DNA,RNA, Hybrid
##################################---------------------------------------------
def main9():
    #### make sure which are sym
    for seq_id in range(17):
        analysis.fit_stencil_in_matrix(DNA.choose_seq([seq_id]),True, "DNA_stiffness_"+str(seq_id+1))
        analysis.fit_stencil_in_matrix(RNA.choose_seq([seq_id]),True, "RNA_stiffness_"+str(seq_id+1))
        analysis.fit_stencil_in_matrix(HYB.choose_seq([seq_id]),False,"HYB_stiffness_"+str(seq_id+1))
        analysis.fit_stencil_in_submatrix(DNA.choose_seq([seq_id]),True, "DNA_stiffness_submatrix_"+str(seq_id+1))
        analysis.fit_stencil_in_submatrix(RNA.choose_seq([seq_id]),True, "RNA_stiffness_submatrix_"+str(seq_id+1))
        analysis.fit_stencil_in_submatrix(HYB.choose_seq([seq_id]),False,"HYB_stiffness_submatrix_"+str(seq_id+1))


##################################---------------------------------------------
# Create groove data and the plot it
##################################---------------------------------------------
def main10():
    # yet to write code for computing it precisely

    # analysis.create_groovewidths_data('DNA')
    # analysis.create_groovewidths_data('RNA')
    # analysis.create_groovewidths_data('HYB')
    # analysis.plot_corr_major_minor()
#    analysis.plot_groovewidths_all()  ### histogram for groove widths
#    analysis.plot_groovewidths_with_seq_condition()
    analysis.plot_groovewidths_seq_logo()   ### yet to check what is going on
    None


def main11():   ######## compare DNA, RNA, etc 
#    analysis.compare_gs_DNA_RNA(DNA,RNA,HYB,'stiff')  ### compare the diagonal element of the stiffness matrix 
    tetramer_points = True
    # analysis.compare_gs_DNA_RNA(DNA,RNA,HYB,'gs',tetramer_points) 
    analysis.compare_gs_DNA_RNA(DNA,RNA,HYB,'stiff',False) 
    return None

def main12():   ### Truncation error
    a1 = analysis.Truncation_error([DNA],['DNA'])
    a2 = analysis.Truncation_error([RNA],['RNA'])
    a3 = analysis.Truncation_error([HYB],['HYB'])
    for i in range(len(a1)):
        print(i+1,' & ',np.around(a1[i],4),' & ',np.around(a2[i],4),' & ',np.around(a3[i],4),'\\\\')
    print(i+1,' & ',np.around(np.mean(a1),4),' & ',np.around(np.mean(a2),4),' & ',np.around(np.mean(a3),4),'\\\\')

    return None
    
def main13():   ### Training error banded to dimer parameters
    a1,a2 = analysis.locality_error(DNA,'DNA','ps2_cgf')
    b1,b2 = analysis.locality_error(RNA,'RNA','ps_rna')
    c1,c2 = analysis.locality_error(HYB,'HYB','ps_hyb')
    for i in range(len(a1)):
        print(i+1,' & ',np.around(a1[i],4)[0][0],' & ',np.around(a2[i],4)[0][0],' & ',np.around(b1[i],4)[0][0], ' & ' ,np.around(b2[i],4)[0][0],' & ',np.around(c1[i],4)[0][0], ' & ' ,np.around(c2[i],4)[0][0],'\\\\')
    print(i+1,' & ',np.around(np.mean(a1),4),' & ',np.around(np.mean(a2),4),' & ',np.around(np.mean(b1),4),' & ',np.around(np.mean(b2),4),' & ',np.around(np.mean(c1),4),' & ',np.around(np.mean(c2),4),'\\\\')

    return None



#main1()  ## comparison of ground state of MD vs reconstruction
#main2()  ## persistence length error
#main3()  ## palindromic error
#main4()  ## training/test error#
#main5()  ## set_scale_for_errors
#main6()  ## compare_same_seq_across_data
#main7()  ## compare_diff_seq_within_data
#main8()  ## plot_eig_olig_first
#main9()   ## fit_stencil_in_matrix
#main10()  ## Create groove data and the plot it
#main11()   #compare_gs_DNA_RNA ..... compare dimer groundstate
#main12()  # compue truncation error for training seq
#main13()  # compute training error .. banded Gussin to model params
####################################################
## Palindrome article code
##################################
#analysis.envelope_KL()


def main_51():    ###### compare two MD and two ps MLE and ME
    ps2_mle_1, ps2_mle_2 = analysis.palin_training_err(DNA,'dna_mle')
    ps2_me_1 , ps2_me_2 = analysis.palin_training_err(DNA,'ps2_cgf')
    ps1_me_1 , ps1_me_2 = analysis.palin_training_err(PDNA,'ps1')

    t1, t2, t3 = analysis.Truncation_error_marg([DNA],['DNA'])
    t4 = analysis.Truncation_error([PDNA],['PDNA'])

    a1,a2 = analysis.compare_MD(DNA,PDNA)
    b1,b2 = analysis.compare_ps(PDNA,'ps2_cgf','dna_mle')

    data = [a1, a2, b1, b2, t1, t2, t3, t4, ps2_mle_1, ps2_mle_2, ps2_me_1, ps2_me_2, ps1_me_1, ps1_me_2]
    label = ['SKL(M1,M2)','SM(M1,M2)','SKL(P2-ME,P2-MLE)','SM(P2-ME,P2-MLE)','Trunc-M2-inter','Trunc-M2-cg','Trunc-M2','Trunc-M1','SKL(M2,P2-MLE)','SM(M2,P2-MLE)','SKL(M2,P2-ME)','SM(M2,P2-ME)','SKL(M1,P1-ME)','SM(M1,P1-ME)']
    analysis.make_table2(data,label)

    path = './BDNA/palin.bscl.tip3p.jc.comb.stats.XXmus.cgF.mat'
    analysis.print_palin_err(path,DNA_217)
    return None

def main_52():  ## print end sequences and hist of sequences
#    analysis.print_end_seq(DNA_ends)
    analysis.hist_seq_training_lib(DNA)
    return None

def main_53():
    for seq_id in range(17):
#        analysis.fit_stencil_in_matrix_3types(DNA.choose_seq([seq_id]),True, "DNA_stiffness_3_types_"+str(seq_id+1))
#        analysis.fit_stencil_in_matrix_3types_combine(DNA.choose_seq([seq_id]),True, "DNA_stiffness_3_types_"+str(seq_id+1))
        None
    analysis.plot_persistence_length_palindrome_article(['DNA_BSTJ_CGF','DNA_BSTJ_MLE'])

def main_54():
    analysis.compare_seq_for_palindrome_article(DNA,'palin_compare')

def main_55():  ####### TTC plots
#    analysis.print_TTC_files(DNA,3,'Unsym_ttc_DNA')  ###only once  
#    analysis.print_TTC_files(DNA,3,'sym_ttc_DNA')  ###only once## modeify loop for 17 seq
    for i in range(17):
        analysis.plot_TTC_files(DNA,'DNA',1+i)
        None
#    analysis.plot_TTC_files_all_ld(DNA,'DNA')

def main_56():  ####### compare two truncation
    for seq_id in range(1):
        analysis.compare_two_trunc(DNA,seq_id,label='DNA')
        None
#    analysis.compare_two_trunc_KL(DNA)        
    return None    

def main57():
    analysis.signals_in_snps()
    analysis.compare_gs_SNiPs()
    return None

def main58():
    for i in range(1,5):
        analysis.plot_Gaussian_error(i)
    return None


def main61():
    analysis.compare_various_errors()
    return None

##################################################################
###################### figures for web article ###################
##################################################################
def web_main1():
    seq_id = 15
    analysis.compare_groundstate_two_paramsets(PDNA, DNA, seq_id)
    return None

##################################################################
##################################################################
##################################################################

web_main1()

#main_51()   #####compare different data set
#main_52()   #### some random stuff
#main_53()   #####fit stencil and persistence length
#main_54()   ########compare sequences
#main_55()   ####### TTC plots
#main_56()    ###### compare_two_trunc


####################################################
## Palindrome article code ----> For RNA  ----9 Feb 2022 -- for thesis
##################################
#main57()
# analysis.compare_seq_for_palindrome_article(DNA,'palin_compare_thesis')  ## yes, I have used this
#main2()
#main58()
#main3()
#main61()
#main10()