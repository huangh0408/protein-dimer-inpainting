from __future__ import division
import numpy as np
from scipy.misc import imsave
import os,sys
#from __future__ import division

def plot(a):
	pdb_name_file="./workspace/pdb_list.txt"
	f=open(pdb_name_file,"r")
	for line in f:
		temp=line.split()
		pdb_name=temp[0]
#		chain_file_name=os.path.join('./workspace/',pdb_name+'_temp_dir/',pdb_name+'_chain.txt')
		chain_file_name=os.path.join('./chain/',pdb_name+'_chain.txt')
		ff=open(chain_file_name,"r")
		for rr in ff:
		#	for ll in ff:			
			temp_1=rr.split()
#			temp_2=ll.split()
			vv=len(temp_1)
			if vv <2:
				continue
			r=temp_1[0]
			l=temp_1[1]
			#if r < l:
			matrix_A_name=os.path.join('./true_contact_matrix/',pdb_name+'_'+r+'_temp'+'.contact_matrix')
			matrix_B_name=os.path.join('./true_contact_matrix/',pdb_name+'_'+l+'_temp'+'.contact_matrix')
			matrix_A_B_name=os.path.join('./true_contact_matrix/',pdb_name+'_'+r+'_'+l+'_temp'+'.contact_matrix')
#			matrix_B_A_name=os.path.join('./true_contact_matrix/',pdb_name+'_'+l+'_'+r+'_temp'+'.contact_matrix')
			exist_A=os.path.exists(matrix_A_name)
			exist_B=os.path.exists(matrix_B_name)
			exist_A_B=os.path.exists(matrix_A_B_name)
			if exist_A==True and exist_B==True and exist_A_B==True:
				size_A=os.path.getsize(matrix_A_name)
				size_B=os.path.getsize(matrix_B_name)
				size_A_B=os.path.getsize(matrix_A_B_name)
				print (pdb_name,r,l)
#			size_B_A=os.path.getsize(matrix_B_A_name)
				if size_A > 0 and size_B > 0 and size_A_B > 0:
					matrix_A=np.loadtxt(matrix_A_name)
					matrix_B=np.loadtxt(matrix_B_name)
					matrix_A_B=np.loadtxt(matrix_A_B_name)
					matrix_A_copy=np.copy(matrix_A)
					matrix_B_copy=np.copy(matrix_B)
					matrix_A_B_copy=np.copy(matrix_A_B)
					for d in range(0,36):
						d_u=2.5+d*0.5
						d_l=2.0+d*0.5
						e1=(matrix_A_copy<=d_u)&(matrix_A_copy>d_l)
						e2=(matrix_B_copy<=d_u)&(matrix_B_copy>d_l)
						e3=(matrix_A_B_copy<=d_u)&(matrix_A_B_copy>d_l)
						matrix_A_copy[e1]=d_l
						matrix_B_copy[e2]=d_l
						matrix_A_B_copy[e3]=d_l
					matrix_A_copy[matrix_A_copy>20]=50
					matrix_A_copy[matrix_A_copy<2]=1.5
#					matrix_A_copy=matrix_A_copy*10
					matrix_A_copy=(100.0/(matrix_A_copy-0.5))+155.0
					matrix_A_copy[matrix_A_copy<160]=0
					matrix_B_copy[matrix_B_copy>20]=50
					matrix_B_copy[matrix_B_copy<2]=1.5
					matrix_B_copy=(100.0/(matrix_B_copy-0.5))+155.0
					matrix_B_copy[matrix_B_copy<160]=0
					matrix_A_B_copy[matrix_A_B_copy>20]=50
					matrix_A_B_copy[matrix_A_B_copy<2]=1.5
#                                       matrix_A_copy=matrix_A_copy*10
					matrix_A_B_copy=(100.0/(matrix_A_B_copy-0.5))+155.0
					matrix_A_B_copy[matrix_A_B_copy<160]=0
                                        
					#	matrix_A_copy[matrix_A_copy<4.5]
				#	matrix_B_copy=np.copy(matrix_B)
				#	matrix_A_B_copy=np.copy(matrix_A_B)
					matrix_A_one_hot=np.copy(matrix_A)
					matrix_A_one_hot[matrix_A_one_hot<=8.0]=1
					matrix_A_one_hot[matrix_A_one_hot>8.0]=0
					matrix_B_one_hot=np.copy(matrix_B)
					matrix_B_one_hot[matrix_B_one_hot<=8.0]=1
					matrix_B_one_hot[matrix_B_one_hot>8.0]=0
					matrix_A_B_one_hot=np.copy(matrix_A_B)
					matrix_A_B_one_hot[matrix_A_B_one_hot<=8.0]=1
					matrix_A_B_one_hot[matrix_A_B_one_hot>8.0]=0
					sum_A_B=matrix_A_B_one_hot.sum()
#					matrix_B_A=np.transpose(matrix_A_B)
					l_A=len(matrix_A)
					l_B=len(matrix_B)
					matrix_B_A=np.zeros((l_B,l_A))
					matrix_B_A_copy=np.zeros((l_B,l_A))
					matrix_B_A_one_hot=np.zeros((l_B,l_A))
					l_A_1=matrix_A_B.shape[0]
					l_B_1=matrix_A_B.shape[1]
					ratio_A_B=sum_A_B/(l_A+l_B)
					l_A_ratio=l_A/(l_A+l_B)
					l_B_ratio=l_B/(l_A+l_B)
					if l_A == l_A_1 and l_B == l_B_1:
#					if l_A == l_A_1 and l_B == l_B_1 and ratio_A_B > 0.1 and l_A>50 and l_B>50 and l_A<500 and l_B<500 and l_A_ratio>0.2 and l_B_ratio>0.2:
						M1=np.hstack((matrix_A,matrix_A_B))
						M2=np.hstack((matrix_B_A,matrix_B))
						flag_A_B=np.vstack((M1,M2))
						M1_one_hot=np.hstack((matrix_A_one_hot,matrix_A_B_one_hot))
						M2_one_hot=np.hstack((matrix_B_A_one_hot,matrix_B_one_hot))
						flag_A_B_one_hot=np.vstack((M1_one_hot,M2_one_hot))
						M1_copy=np.hstack((matrix_A_copy,matrix_A_B_copy))
						M2_copy=np.hstack((matrix_B_A_copy,matrix_B_copy))
						flag_A_B_copy=np.vstack((M1_copy,M2_copy))
						flag_A_B_matrix_name_dis_true=os.path.join('./flag_distance_matrix_true/',pdb_name+'_'+r+'_'+l+'.txt')
						flag_A_B_image_name=os.path.join('./flag_contact_image/',pdb_name+'_'+r+'_'+l+'.jpg')
						flag_A_B_length_name=os.path.join('./flag_length/',pdb_name+'_'+r+'_'+l+'.txt')
						flag_A_B_matrix_name_dis=os.path.join('./flag_distance_matrix_multi/',pdb_name+'_'+r+'_'+l+'.txt')
						flag_A_B_image_name_dis=os.path.join('./flag_distance_image/',pdb_name+'_'+r+'_'+l+'.jpg')
#                                                flag_A_B_length_name_dis=os.path.join('./flag_distance_length/',pdb_name+'_'+r+'_'+l+'.txt')
						f1=open(flag_A_B_length_name,"w")
						print >> f1, "%d %d" % (l_A, l_B)
#						f2=open(flag_A_B_le_name,"w")
						imsave(flag_A_B_image_name,flag_A_B_one_hot)
						imsave(flag_A_B_image_name_dis,flag_A_B_copy)
						np.savetxt(flag_A_B_matrix_name_dis_true,flag_A_B)
						np.savetxt(flag_A_B_matrix_name_dis,flag_A_B_copy)
		


if __name__=="__main__":
	
	a=1
	plot(a)

