#!/usr/bin/env python
import numpy as np
import os,sys
#from utils.acc_cal_v2 import topKaccuracy, evaluate, output_result
from utils.acc_cal_for_interaction import topKaccuracy_temp, evaluate_temp, output_result_temp,output_result_number,evaluate_mae,save_distance
from PIL import Image
#import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
import skimage.io
import skimage.transform
import argparse

#def evaluate_tmp(predict_matrix,ccmpred_matrix,contact_matrix):
#input_acc=[]
output_acc=[]
mae_acc=[]
#DeepCov_acc=[]
#psicov_acc=[]
#ccmpred_acc=[]
j=0
#date=sys.argv[1]

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", default="./results", help="where to put output files")
parser.add_argument("--ground_truth_dir", default="./datasets", help="where to put output files")
parser.add_argument("--mode", required=True, choices=["contact", "distance", "slice"])
parser.add_argument("--type", required=True, choices=["homodimer", "heteromer", "all"])
parser.add_argument("--netsize", type=int,required=True, choices=[128, 256, 512])
parser.add_argument("--interation", type=int, default=100, help="number of images in batch")
hh = parser.parse_args()
output_name=hh.output_dir
groundtruth_name=hh.ground_truth_dir
dataset_type=hh.mode
dataset_source=hh.type
net_size=hh.netsize
inter=hh.interation
#predicted_distance_dir='./predictions_distance/'
predicted_rr_dir='./masif_homo_predictions_rr/'
if not os.path.exists(predicted_rr_dir):
            os.makedirs( predicted_rr_dir )
result_path= os.path.join(output_name,'test_'+dataset_source+'_'+dataset_type+'_'+str(net_size))
output_dir=os.path.join(output_name,dataset_source+'_'+dataset_type+'_'+str(net_size))
length_dir_temp=os.path.join(groundtruth_name,'length')
truth_dir=os.path.join(groundtruth_name,'matrix')
pdb_name_file="cd-hit-pdb_homo_list.txt"
f=open(pdb_name_file,"r")
ii=0
for line in f:
    temp=line.split()
#                pdb_name=temp[0]
#for filename in os.listdir(output_dir):
#	temp=filename.split('.')
    name_img=temp[0]
    name_temp=temp[0].split('_')
#	name=name_temp[1]+'_'+name_temp[2]+'_'+name_temp[3]
    name=name_img
    length_dir=os.path.join(length_dir_temp,name+'.txt')
    true_matrix_name=os.path.join(truth_dir,name+'.txt')
#	fasta_file=os.path.join('/home/huanghe/huangh/inter_protein_contact_prediction_data_prepare/dc_train/feature_contact_matrix/ccmpred/',name+'.ccmpred')
#	temp_matrix=np.loadtxt(fasta_file)
#	ccmpred_matrix=temp_matrix
    if ii >=0:
        try:   
            data_1=np.loadtxt(length_dir)
            data_2=np.loadtxt(true_matrix_name)
        except Exception as e:
            print("fail to open file %s" %name)
#                       pass
            continue
        length=np.loadtxt(length_dir)
        L1=length[0]
        L2=length[1]
        l1=int(L1)
        l2=int(L2)
        L=l1+l2
#	L=temp_matrix.shape[1]
#	ccmpred_matrix_name=fasta_file
#	DeepCov_matrix_name=os.path.join('./feature_2_21/feature_contact_matrix/DeepCov',name+'.matrix')
#	psicov_matrix_name=os.path.join('./feature_2_21/feature_contact_matrix/psicov',name+'.matrix')
#	true_image_name=os.path.join('/home/huanghe/huangh/GAN-collection/image-inpainting/Inpainting/workspace_3_19/true_dir',name+'.txt')
        output_image_name=os.path.join(output_dir,'img_'+str(ii)+'.'+str(inter)+'.jpg')
#	predict_matrix_file=os.path.join(output_dir,name+'-outputs.png')
#	ccmpred_matrix=os.path.join('./TEST.1.7/images/',name+'-inputs.png')
#	contact_matrix_file=os.path.join('./hh_2_21_1/images/',name+'-targets.png')
#	true_matrix_file_gray=Image.open(true_image_name).convert('L')
#	true_matrix_file_temp=misc.imresize(true_matrix_file_gray,[L,L],interp='nearest')
#	true_matrix=np.array(true_matrix_file_temp)
        output_matrix_file_gray=Image.open(output_image_name).convert('L')
        output_matrix_temp=np.array(output_matrix_file_gray)
        output_matrix=skimage.transform.resize(output_matrix_temp,[L,L])
#        output_matrix_file_temp=misc.imresize(output_matrix_file_gray,[L,L],interp='nearest')
#        output_matrix=np.array(output_matrix_file_temp)
#	true_sub_matrix=true_matrix[:l1,l1:L]
        output_sub_matrix=output_matrix[:l1,l1:L]
        true_matrix_name=os.path.join(truth_dir,name+'.txt')
#        true_matrix_gray=Image.open(true_matrix_name).convert('L')
 #       true_matrix_temp=misc.imresize(true_matrix_gray,[L,L],interp='nearest')
#	true_matrix_temp=skimage.io.imread(true_matrix_name)
        true_matrix=np.loadtxt(true_matrix_name)
        true_sub_matrix=true_matrix[:l1,l1:L]
        ii+=1
#	true_sub_matrix_temp[true_sub_matrix_temp<1]=1
 #       true_sub_matrix=255.0/true_sub_matrix_temp
#	output_sub_matrix_temp[output_sub_matrix_temp<1]=1
#	output_sub_matrix=255.0/output_sub_matrix_temp
	
#	matrix_A_B_name=os.path.join('/home/huanghe/huangh/bioinfo_hh/Complex_Tool/contact_1_14/',pdb_name+'_'+chain_A+'_'+chain_B+'.contact_matrix')
#	contact_matrix_file_temp=misc.imresize(contact_matrix_file,[L,L],interp='nearest')
#	contact_matrix_file_temp
#	im1=Image.open(predict_matrix_file_temp)
#	im1=predict_matrix_file_temp
#	im2=Image.open(contact_matrix_file_temp)
#	predict_matrix=np.array(predict_matrix_file_temp)
#	predict_name=os.path.join('/home/huanghe/huangh/bioinfo_hh/deepcontact/result_3_10/',name+'.txt')
	#predict_matrix=np.loadtxt(predict_name)
#	predict_matrix=np.loadtxt(predict_name)
#	contact_matrix=np.array(im2)
#	contact_matrix=np.loadtxt(contact_matrix_name)
	
#	DeepCov_matrix=np.loadtxt(DeepCov_matrix_name)
#	psicov_matrix=np.loadtxt(psicov_matrix_name)
#	DeepCov_acc.append(evaluate(DeepCov_matrix, contact_matrix))
#	psicov_acc.append(evaluate(psicov_matrix, contact_matrix))
#	ccmpred_acc.append(evaluate(ccmpred_matrix, contact_matrix))
#	input_acc.append(evaluate(_matrix, contact_matrix))
#        output_sub_matrix[output_sub_matrix<8]=1
 #       output_sub_matrix[output_sub_matrix==8]=1
  #      output_sub_matrix[output_sub_matrix>8]=0
#        hhh=evaluate_mae(output_sub_matrix,true_sub_matrix)
#	true_sub_matrix=true_sub_matrix/255.0
        true_sub_matrix[true_sub_matrix<8]=1
        true_sub_matrix[true_sub_matrix>=8]=0
        output_sub_matrix=output_sub_matrix/255.0
 #       save_distance(output_sub_matrix,true_sub_matrix,predicted_rr_dir,name)
#        f_t=open(predicted_distance_dir+name+'_t.txt','w')
 #       np.savetxt(f_t,true_sub_matrix)
  #      f_t.close()
   #     f_p=open(predicted_distance_dir+name+'_p.txt','w')
    #    np.savetxt(f_p,output_sub_matrix)
     #   f_p.close()
#        true_sub_matrix[true_sub_matrix<31.25]=0
#        true_sub_matrix[true_sub_matrix==8]=1
 #       true_sub_matrix[true_sub_matrix>=31.25]=1
        hhh=1
        gap=abs(l1-l2)
        
        #print(gap)
       # print("\nAcc result:%s" %name)
        try:
            tt=evaluate_temp(output_sub_matrix,true_sub_matrix)
        except:
		#pass
                continue
        #print(tt[8])
#        if tt[8]>0 and gap==0 and L <512 and L >128:
        if tt[8]>0:
#            save_distance(output_sub_matrix,true_sub_matrix,predicted_rr_dir,name)
            output_acc.append(evaluate_temp(output_sub_matrix,true_sub_matrix))
            mae_acc.append(hhh)
#	print "\n"
#	print "*"*50
#		print "\nAcc result:%s" %name
#	print "*"*50
#	print "\nccmpred result accuracy:" 
#        output_result(ccmpred_acc[j])
#	print "\npsicov result accuracy:"
 #       output_result(psicov_acc[j])
#	print "\nDeepCov result accuracy:"
 #       output_result(DeepCov_acc[j])
           # print("\nOutput result accuracy:") 
           # output_result_temp(output_acc[j],mae_acc[j])
#	print "\nDeepCov result:"
#        output_result(DeepCov_acc[j])
            j+=1
#
#print "Input result:"
#output_result(np.mean(np.array(input_acc), axis=0))
print("\n"*5)
print("*"*50)
#print "\nccmpred total result:"
#print "*"*50
#output_result(np.mean(np.array(ccmpred_acc), axis=0))
#print "\n"*5
#print "*"*50
#print "\nDeepCov total result:"
#output_result(np.mean(np.array(DeepCov_acc), axis=0))
#print "\n"*5
#print "*"*50
#print "\npsicov total result:"
#output_result(np.mean(np.array(psicov_acc), axis=0))
#print "\n"*5
#print "*"*50
print("\noutput total result accuracy:")
print(inter)
output_result_temp(np.mean(np.array(output_acc), axis=0),np.mean(np.array(mae_acc), axis=0))
print("\n"*5)
print("*"*50)
print("\noutput total result number:")
output_result_number(np.array(output_acc))
