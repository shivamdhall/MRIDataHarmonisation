'''
Please note the functions below are specially designed to extract relevant data from 
raw DWMRI data that was provided for purposes of this project only.  The below functions can 
be used to create directory structures, clean raw data, separate DW-MRI data based on b-values
and combine b-volumes into a single DW-MRI dataset. 

NOTE: These functions are not guaranteed to generalise to other datasets.
This is simply due to the various differences that exist between DW-MRI datasets, these may include:
the protocol used to collect data, the range of b-values used, the structure of the stored dataset, 
naming convenetions, the amount of preprocessing applied to the dataset, the type of scanner used to acquire the data.
'''

import os
import shutil
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Clean up raw MRI data.')
parser.add_argument('--clean_raw_data', required=False, nargs=2, metavar=('RAW_PATH', 'STORE_PATH'), help='Arg1: Path to directory containg raw DW-MRI data, Arg2: Path to directory containing folder structure')
parser.add_argument('--create_structure', required=False, nargs=2, metavar=('PATH', 'NAME'), help='Arg1: Directory to create folder structue, Arg2: Dataset name')
parser.add_argument('--separate_data', required=False, metavar='PATH', help='Arg1: Directory contatining data to seperate')
parser.add_argument('--combine_volumes', required=False, metavar='PATH', help='Arg1: Path of directory containing seperated data')
params = parser.parse_args()
print(params)

if params.separate_data:
    #create a separated directory for the cleaned data scans - to save space
    os.chdir(params.separate_data)
    for direc in os.listdir(os.getcwd())[1:]:
        os.chdir('./' + direc)
        if not os.path.exists('Data_separated'):
            os.makedirs('Data_separated')
        os.chdir('./Data_separated')
        if not os.path.exists('b_300'):
            os.makedirs('b_300')
        if not os.path.exists('b_700'):
            os.makedirs('b_700')
        if not os.path.exists('b_2000'):
            os.makedirs('b_2000')
        os.chdir(params.separate_data + '/' + direc)
        if os.path.exists('DFC'):
            os.chdir('./DFC')
            files = os.listdir(os.getcwd())
            shutil.copy(files[0] , '../Data_separated/b_700')
            shutil.copy(files[1] , '../Data_separated/b_700')
            shutil.copy(files[3] , '../Data_separated/b_2000')
            shutil.copy(files[4] , '../Data_separated/b_2000')
            shutil.copy(files[6] , '../Data_separated/b_300')
            shutil.copy(files[7] , '../Data_separated/b_300')
            os.chdir('../ORIG')
            files = os.listdir(os.getcwd())
            shutil.copy(files[2] , '../Data_separated/b_700')
            shutil.copy(files[5] , '../Data_separated/b_2000')
            shutil.copy(files[8] , '../Data_separated/b_300')
            os.chdir('..')
            shutil.rmtree('DFC')
            shutil.rmtree('ORIG')
        os.chdir(params.separate_data)    

if params.create_structure:
    #create folder structure for dataset
    os.chdir(params.create_structure[0])
    if not os.path.exists(params.create_structure[1]):
        os.makedirs(params.create_structure[1])
        os.chdir('./' + params.create_structure[1])
        for i in range(1, 11):
            for j in range(1, 3):
                dir_name = 'Subj' + str(i) + 'Scan' + str(j)
                os.makedirs(dir_name)
        for direc in os.listdir(os.getcwd()):
            os.chdir('./' + direc)
            if not os.path.exists('Data_separated'):
                os.makedirs('Data_separated')
            os.chdir('./Data_separated')
            if not os.path.exists('b_300'):
                os.makedirs('b_300')
            if not os.path.exists('b_700'):
                os.makedirs('b_700')
            if not os.path.exists('b_2000'):
                os.makedirs('b_2000')
            os.chdir(params.create_structure[0] + '/' + params.create_structure[1])
        #directories for each subject/scan have been made

if params.clean_raw_data:
    #clean up data
    #populating Data_separated folders
    scan1_dfc = [8, 10, 12] #used for bvec and bval 
    scan1_orig = [7, 9, 11] #used for the niFti data
    scan2_dfc = [20, 22, 24]
    scan2_orig = [19, 21, 23]
    dfc = [scan1_dfc, scan2_dfc]
    orig = [scan1_orig, scan2_orig]
    subj = [1,2,3,4,5,6,7,8,9,10]
    bval_ordering = [2000, 300, 700]
    subj_iter = 0
    os.chdir(params.clean_raw_data[0])
    for direc in os.listdir(os.getcwd())[1:]:
        os.chdir('./' + direc)
        print os.getcwd()
        direc2 = os.listdir(os.getcwd())[-1]
        os.chdir('./' + direc2 + '/scans')
        for scan in range(0,2):
            bval_iter = 0
            #for loop for bvals and bvecs
            for file_iterator in dfc[scan]:
                os.chdir('./' + str(file_iterator) + '/DICOM')
                files = os.listdir(os.getcwd())
                shutil.copy(files[-2] , params.clean_raw_data[1] + '/Subj'+str(subj[subj_iter])+'Scan'+str(scan+1)+'/Data_separated/b_'+str(bval_ordering[bval_iter]))
                shutil.copy(files[-3] , params.clean_raw_data[1] + '/Subj'+str(subj[subj_iter])+'Scan'+str(scan+1)+'/Data_separated/b_'+str(bval_ordering[bval_iter]))
                os.chdir('../..')
                bval_iter += 1
            #for loop for niFti data
            bval_iter = 0
            for file_iterator in orig[scan]:
                os.chdir('./' + str(file_iterator) + '/DICOM')
                files = os.listdir(os.getcwd())
                shutil.copy(files[-1] , params.clean_raw_data[1] + '/Subj'+str(subj[subj_iter])+'Scan'+str(scan+1)+'/Data_separated/b_'+str(bval_ordering[bval_iter]))
                os.chdir('../..')
                bval_iter += 1
        subj_iter += 1
        os.chdir(params.clean_raw_data[0])


if params.combine_volumes:
    #Create combined nifti dataset for all bvals/bvecs
    #also create combined bvec and bval files
    os.chdir(params.combine_volumes)
    #first create the bval and the bvec files
    bval_list = [700, 2000, 300]
    b300_nifti_file = ''
    for direc in os.listdir(os.getcwd())[1:]:
        final_bvals = []
        final_bvecs_x = []
        final_bvecs_y = []
        final_bvecs_z = []
        os.chdir('./' + direc + '/Data_separated')
        for bval in bval_list:
            os.chdir('./b_' + str(bval))
            files = os.listdir(os.getcwd())
            bval_file = open(files[-2],"r")
            bvals = bval_file.readlines()
            bvals = bvals[0].split()
            
            bvec_file = open(files[-1], "r")
            bvecs = bvec_file.readlines()
            bvecs_x = bvecs[0].split()
            bvecs_y = bvecs[1].split()
            bvecs_z = bvecs[2].split()
            
            if bval == 700 or bval == 2000:
                bvals = bvals[:-1]
                bvecs_x = bvecs_x[:-1]
                bvecs_y = bvecs_y[:-1]
                bvecs_z = bvecs_z[:-1]
                if bval == 700:
                    subprocess.call(["fslroi", files[-3], "temp.nii.gz", "0", "36"])
                if bval == 2000:
                    subprocess.call(["fslroi", files[-3], "temp.nii.gz", "0", "72"])
            else:
                #store handel to niFti file for b_300
                b300_nifti_file = files[-3]
                  
            final_bvals = final_bvals + bvals
            final_bvecs_x = final_bvecs_x + bvecs_x
            final_bvecs_y = final_bvecs_y + bvecs_y
            final_bvecs_z = final_bvecs_z + bvecs_z
            
            os.chdir('..')
            
        os.chdir('..')
        #write the final bval/bvec files here
        bval_str = " ".join(final_bvals)
        bvec_str = " ".join(final_bvecs_x) + '\n' + " ".join(final_bvecs_y) + '\n' + " ".join(final_bvecs_z)

        bval_write_file = open('NODDI.bval','w')
        bval_write_file.write(bval_str)
        #close the file to ensure buffers are flushed (write takes place)
        bval_write_file.close()
        
        bvec_write_file = open('NODDI.bvec','w')
        bvec_write_file.write(bvec_str)
        bvec_write_file.close()
        
        subprocess.call(["fslmerge", "-a", "NODDI_3Shells.nii.gz", "./Data_separated/b_700/temp.nii.gz", "./Data_separated/b_2000/temp.nii.gz", "./Data_separated/b_300/" + b300_nifti_file])
        
        #recursively remove the temp files
        print os.getcwd()
        os.chdir('./Data_separated')
        for bval in bval_list:
            os.chdir('./b_' + str(bval))
            if os.path.exists('temp.nii.gz'):
                os.remove('temp.nii.gz')
            os.chdir('..')
        
        os.chdir(params.combine_volumes)    
            
