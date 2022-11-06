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
c4 = ['maroon','dodgerblue','limegreen']

#################### Path for 10 mus final files
DNA_path    = './BDNA/palin.bscl.tip3p.jc.comb.sym.stats.10mus.cgF.25_tot.mat'
MDNA_path   = './BDNA/methyl.bscl.tip3p.jc.comb.sym.stats.10mus.cgF.comb.modified.mat'    
HDNA_path   = './BDNA/hmethyl.bscl.tip3p.jc.comb.modified.sym.stats.10mus.cgF.mat'

DNA_path_3  = './BDNA/palin.bscl.tip3p.jc.comb.sym.stats.10mus.cgF.26_28.mat'
MDNA_path_3 = './BDNA/methyl.bscl.tip3p.jc.comb.sym.stats.10mus.30_32.modified.mat'    
HDNA_path_3 = './BDNA/hmethyl.bscl.tip3p.jc.comb.sym.stats.10mus.30_32.modified.mat'
############### Initiate MD data #################################
DNA = init_MD_data().load_data(DNA_path)
MDNA = init_MD_data().load_data(MDNA_path)
HDNA = init_MD_data().load_data(HDNA_path)

DNA_3 = init_MD_data().load_data(DNA_path_3)
MDNA_3 = init_MD_data().load_data(MDNA_path_3)
HDNA_3 = init_MD_data().load_data(HDNA_path_3)

############### comparison of ground state of MD vs reconstruction #################################
def main1():
    analysis.plot_gs_vs_MD_shape_epi(MDNA,HDNA)

############### persistence lengths plots #################################
def main2():
    p1 = '/Users/rsharma/Dropbox/PhD_work/MD_analysis/persis_len/'
#    analysis.map_seq_file_list_epi_persistence() 
    data = analysis.load_persistence_length_data_epi(p1)

############### palindromic error #################################
def main3():
    path = './BDNA/methyl.bscl.tip3p.jc.comb.sym.stats.XXmus.cgF.mat'
    analysis.plot_heatmap_palin_err_epi(path,name='MDNA')
    path = './BDNA/hmethyl.bscl.tip3p.jc.comb.sym.stats.XXmus.cgF.mat'
    analysis.plot_heatmap_palin_err_epi(path,name='HDNA')

############### training/test error #################################
def main4():
    analysis.plot_heatmap_training_err_epi(MDNA,MDNA_3,'ps_mdna')
    analysis.plot_heatmap_training_err_epi(HDNA,HDNA_3,'ps_hdna')

############### set scale for training/test/palindromic error #################################
def main5():
    analysis.set_scale_for_error_epi(MDNA,sym=True)
    analysis.set_scale_for_error_epi(HDNA,sym=True)

############### compare_same_seq_across_data #################################
linestyles= ['-','-', '--', '-.', ':']
def main6():
    col = [c1,c4,c4]
    for sequence_id in analysis.MDNA_map:
        analysis.compare_same_seq_across_data_epi(MDNA,HDNA,DNA,col,sequence_id,'compare_seq_MDNA_HDNA')
    
############### compare_same_seq_across_data #################################
linestyles= ['-','-', '--', '-.', ':']
## cpg islands comparison
def main7():
    col = [c1,c4,c4]
    analysis.compare_cpg_island_seq_across_data_epi(DNA_3,MDNA_3,HDNA_3,col,'cpg_islands_')


##################################---------------------------------------------
# plot oligomer level eigenvalues for DNA,RNA, Hybrid
##################################---------------------------------------------

def main8():
    #### make sure which are sym
    sym=[True,True,True]
    for enum, seq_id in enumerate(analysis.MDNA_map[0:12]):
        seq = analysis.unmodify(MDNA.seq[seq_id])
        res = cgDNA(seq,'ps2_cgf')
        analysis.plot_eig_olig_first_epi(res, MDNA.choose_seq([seq_id]), HDNA.choose_seq([seq_id]), sym, c1, "Epi_olig_eig_"+str(1+enum))


##################################---------------------------------------------
# plot oligomer level stiffness for DNA,RNA, Hybrid
##################################---------------------------------------------
def main9():
    #### make sure which are sym
    for enum, seq_id in enumerate(analysis.MDNA_map[0:12]):
        analysis.fit_stencil_in_matrix(MDNA.choose_seq([seq_id]),True, "MDNA_stiffness_"+str(seq_id+1))
        analysis.fit_stencil_in_matrix(HDNA.choose_seq([seq_id]),True, "HDNA_stiffness_"+str(seq_id+1))


##################################---------------------------------------------
# Truncation error
##################################---------------------------------------------

def main10():
        data = [DNA, MDNA, HDNA]
        NA_type_list = ['DNA', 'MDNA', 'HDNA']
        analysis.Truncation_error(data, NA_type_list)
        analysis.locality_error_epi(data, NA_type_list)
##################################---------------------------------------------
# Positive def reconstruction tests --- reconstruct seq for given length 
##################################---------------------------------------------
def main11():   
    analysis.check_pos_def_prmset('ps_mdna',False,'MDNA')    
    # analysis.check_pos_def_prmset('ps_hdna',False,'MDNA')

##################################---------------------------------------------
# Analysis hydroxy/methylation effect on GC islands
##################################---------------------------------------------
def main12():
    analysis.stiffness_analysis_MDNA(DNA_3,MDNA_3,HDNA_3)

def main13():   ######## compare DNA, MDNA, etc 
    tetramer_points = False
    analysis.compare_gs_DNA_epi(DNA,MDNA,HDNA,'stiff',tetramer_points)  ### compare the diagonal element of the stiffness matrix 
    analysis.compare_gs_DNA_epi(DNA,MDNA,HDNA,'gs',tetramer_points) 
    return None


def main14():
    #### nbr_seq, GC_content, seq_length, nbr_CpG, save_name
    ## all the scripts need to be done only once 
    create_seq_file        = False
    methylate_artifical    = False
    methylate_human_genome = True


    ### first script create random cpg and not cpg islands 
    if create_seq_file == True:  ## 7 
        print("create_random_cpg_islands")
        nbr_seq = 20000
        analysis.create_random_cpg_islands(nbr_seq, 0.505, 220, 13, 'Sublist_1')  ## 13/220 = 6 %
        analysis.create_random_cpg_islands(nbr_seq, 0.550, 220, 20, 'Sublist_2')  ## 20/220 = 9%
        analysis.create_random_cpg_islands(nbr_seq, 0.550, 220, 26, 'Sublist_3')  ## 26/220 = 12%
        analysis.create_random_cpg_islands(nbr_seq, 0.650, 220, 33, 'Sublist_4')  ## 35/220 = 15%
        analysis.create_random_cpg_islands(nbr_seq, 0.700, 220, 39, 'Sublist_5')  ## 39/220 = 18%
        analysis.create_random_not_cpg_islands(nbr_seq, 220, 6, 'Sublist_6') ##### 2.5%  ## Not CpG
        analysis.create_random_not_cpg_islands(nbr_seq, 220, 3, 'Sublist_7') ### .12%%   ## Not CpG

    ### methylate randomly created CpG isalnds
    if methylate_artifical == True:
        print("methylate_artifical created cpg isalnds")
        for s in range(1,6): ## 60
            for p,i in zip([0.25,0.50,0.75,1],[1,2,3,4]):
                analysis.methylate_files('Sublist_'+str(s)+'_0' ,p, ['MN'],      str(i)+'_MN')
                analysis.methylate_files('Sublist_'+str(s)+'_0' ,p, ['MG'],      str(i)+'_MG')
                analysis.methylate_files('Sublist_'+str(s)+'_0' ,p, ['MG','MN'], str(i)+'_MG_MN')

        for s in range(6,8):  ##12
            for p,i in zip([0.50,1],[2,4]):
                analysis.methylate_files('Sublist_'+str(s)+'_0' ,p, ['MN'],      str(i)+'_MN')
                analysis.methylate_files('Sublist_'+str(s)+'_0' ,p, ['MG'],      str(i)+'_MG')
                analysis.methylate_files('Sublist_'+str(s)+'_0' ,p, ['MG','MN'], str(i)+'_MG_MN')

    ## read the CpG islands files provided by Daiva and reclassify them as per criteria
    ## modify them accordingly
    if methylate_human_genome == True:  

        print("methylate human  cpg isalnds")
        analysis.read_and_rewrite_cpg_island_data()
        for p,i in zip([0.25,0.50,0.75,1],[1,2,3,4]):  ##12
            analysis.methylate_files('Human_CpG_islands_modified_0'     ,p, ['MN'],      str(i)+'_MN')
            analysis.methylate_files('Human_CpG_islands_modified_0'     ,p, ['MG'],      str(i)+'_MG')
            analysis.methylate_files('Human_CpG_islands_modified_0'     ,p, ['MG','MN'], str(i)+'_MG_MN')
        for p,i in zip([0.50,1],[2,4]):   ## 6
            analysis.methylate_files('Human_NOT_CpG_islands_modified_0' ,p, ['MN'],      str(i)+'_MN')
            analysis.methylate_files('Human_NOT_CpG_islands_modified_0' ,p, ['MG'],      str(i)+'_MG')
            analysis.methylate_files('Human_NOT_CpG_islands_modified_0' ,p, ['MG','MN'], str(i)+'_MG_MN')
        
    return None


def main15():
    ## idea is to perform sensitivity analysis on the groundstate on DNA on the methylation of central CpG steps
    for metric in ['Mahal']: #, 'abs'
        analysis.plot_seq_logo_epi_sensitivity('MN','ps_mdna',metric)
        analysis.plot_seq_logo_epi_sensitivity('MG','ps_mdna',metric)
        analysis.plot_seq_logo_epi_sensitivity('HK','ps_hdna',metric)
        analysis.plot_seq_logo_epi_sensitivity('HG','ps_hdna',metric)
    return None

def main16():
    #compare_epi_sensitivity_on_groundstate(s1,s2,modi,modi_pos,ps,save_name)
    s1 = 'GCGTCGGTA'+'ACGT'+'TTTGTCGGC'
    s2 = 'GCGTCGGTT'+'GCGC'+'TTTGTCGGC'
    modi,modi_pos = 'MN', 10
    analysis.compare_epi_sensitivity_on_groundstate(s1,s2,modi,modi_pos,'ps_mdna',"compare_epi_sensitivity_on_groundstate_")
    return None

def main17():
#    analysis.create_groovewidths_data_epi('MDNA','ps_mdna')
#    analysis.create_groovewidths_data_epi('HDNA','ps_hdna')
#    analysis.plot_groovewidths_data_epi_cpg()
    analysis.plot_groovewidths_data_epi_cpg2('minor_diff')
    analysis.plot_groovewidths_data_epi_cpg2('major_diff')
    return None

def main18():
    analysis.seq_logo_demo()
    return None

#main1()  ## comparison of ground state of MD vs reconstruction
#main2()  ## persistence length error
#main3()  ## palindromic error
#main4()  ## training/test error, reconstruction error
#main5()  ## set_scale_for_errors
#main6()  ## compare_same_seq_across_data
#main7()  ## compare_ cpg island sequences
#main8()  ## plot_eig_olig_first
#main9()   ## fit_stencil_in_matrix
#main10()  ### Truncation error and locality error 
# analysis.mdna_random_seq(1)    #########not completed this yet to do
main13()     ### compare_gs_DNA_epi
#main14()   ##### create_random_cpg_islands  and modify human cpg islands
#main15()   ######sequence -logo plot for sensitivity analysis
#main16()
#main17()     ### groove width analysis
#main18()  ### plot for seq logo demostration
