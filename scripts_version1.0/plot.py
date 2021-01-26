import os
from PIL import Image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.misc as misc
import skimage.io
from matplotlib.pyplot import figure

def draw_image(name_list):
	f=open(name_list,'r')
	lines=f.readlines()
	f.close()
	j=1
	for line in lines:
		line=line.strip("\n")
		name=line.split(".")[0]
		jpgname=os.path.join('./jpg/',name+'.png')
#		fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(12,5))
		true_dir=os.path.join('/extendplus/huanghe/dimer_workspace/3dcomplex-dimer/feature_workspace_12_29/flag_distance_matrix_true/',name+'.txt')
		true_multi_dir=os.path.join('/extendplus/huanghe/dimer_workspace/3dcomplex-dimer/feature_workspace_12_29/flag_distance_matrix_multi/',name+'.txt')
		pred_dir=os.path.join('/extendplus/huanghe/dimer_workspace/model_training/workspace_contact/output_dir_12_24/',name+'.jpg')
		length_dir=os.path.join('/extendplus/huanghe/dimer_workspace/3dcomplex-dimer/feature_workspace_12_29/flag_length',name+'.txt')
#		name_4=os.path.join('./rmsd_temp/',name+'_choose_predict_rmsd.txt')
		try:
			data_1=np.loadtxt(true_dir)
			data_2=np.loadtxt(true_multi_dir)
			data_3=np.loadtxt(length_dir)
#			data_4=np.loadtxt(name_4)
		except Exception as e:
			print("fail to open file %s" %name)
#			pass
			continue
#		fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(12,5))
		length=np.loadtxt(length_dir)
		L1=length[0]
		L2=length[1]
		l1=int(L1)
		l2=int(L2)
		L=l1+l2
		output_matrix_file_gray=Image.open(pred_dir).convert('L')
		output_matrix_file_temp=misc.imresize(output_matrix_file_gray,[L,L],interp='nearest')
		output_matrix=np.array(output_matrix_file_temp)
		output_sub_matrix=output_matrix[:l1,l1:L]
		true_matrix=np.loadtxt(true_dir)
		true_multi_matrix=np.loadtxt(true_multi_dir)
		true_sub_matrix=true_matrix[:l1,l1:L]
		figure1_matrix=np.copy(true_matrix)
		zero_matrix=np.zeros((l1,l2))
		figure1_matrix[figure1_matrix<8]=1
		figure1_matrix[figure1_matrix>=8]=0
		figure2_matrix=np.copy(figure1_matrix)
		figure3_matrix=np.copy(figure1_matrix)
		figure4_matrix=np.loadtxt(true_multi_dir)
		figure4_matrix=figure4_matrix/255.0
		figure1_matrix[:l1,l1:L]=zero_matrix
		figure1_matrix[l1:L,0:l1]=np.transpose(zero_matrix)
		figure2_matrix[l1:L,0:l1]=np.transpose(zero_matrix)
		output_sub_matrix=output_sub_matrix/255.0
		figure2_matrix[:l1,l1:L]=output_sub_matrix
		figure3_matrix[l1:L,0:l1]=np.transpose(figure3_matrix[0:l1,l1:L])
		figure4_matrix[l1:L,0:l1]=np.transpose(figure4_matrix[0:l1,l1:L])
		p_dict={}
		for m in range(0,l1):
			for n in range(0,l2):
				p_dict[(m,n)]=output_sub_matrix[m,n]
			#	y_dict[(m,n)]=true_sub_matrix[m,n]
		x=0
		flag_index=np.ones((L,2))
		flag_true=np.ones((L,2))
		xx=0
		yy=0
		first_contact=L+1
		all_contact=np.sum(figure3_matrix[0:l1,l1:L])
		contact_temp1=all_contact/(L+0.0)
		contact_density = "%.2f%%" % (contact_temp1 * 100)
		for pair in sorted(p_dict.items(), key=lambda x: x[1],reverse=True):
			(k, v) = pair
			(i,j)=k
#			x+=1
#			flag[
			if x == L:
				break
                 #       flag_index[x,0]=i
		#	flag_index[x,1]=j+l1
			if figure3_matrix[i,j+l1] == 1:
				flag_true[xx,0]=i
				flag_true[xx,1]=j+l1
				xx+=1	
				if x < first_contact:
					first_contact=x+1
			else:
				flag_index[yy,0]=i
	                        flag_index[yy,1]=j+l1
				yy+=1
			x+=1
		pos=xx
		neg=yy
		contact_temp=xx/(L+0.0)
		contact_precision = "%.2f%%" % (contact_temp * 100)
#		contact_precision=xx/(L+0.0)
		flag_neg=np.ones((neg,2))
		flag_pos=np.ones((pos,2))
		flag_neg=flag_index[0:neg,:]
		flag_pos=flag_true[0:pos,:]
#		flag_value=
		X=np.linspace(0,L,30)
		Y=np.ones((30,1))*l1
		title_1='Input'
		title_2='Output'
		title_3='Cont Map'
		title_4='Dist Map'
		figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', frameon=True, edgecolor='k')
		plt.suptitle(name)
		plt.subplot(2,2,1)
	#	plt.grid(None)
		plt.imshow(figure1_matrix,cmap='binary')
		plt.plot(X,Y,c='b')
		plt.plot(Y,X,c='b')
		plt.title(title_1,y=-0.1)
		x_label=l1/3
		y_label=l2/2+l1
		chain_info='l1: '+str(l1)+' ; '+'l2: '+str(l2)+' ; '+'L: '+str(L)
		plt.text(x_label,y_label,chain_info)
		plt.subplot(2,2,2)
         #       plt.grid(None)
                plt.imshow(figure2_matrix,cmap='binary')
		plt.plot(X,Y,c='b')
                plt.plot(Y,X,c='b')
		plt.title(title_2,y=-0.1)	
		x_label=l1/4
                y_label=l2/2+l1
		contact_info='first: '+str(first_contact)+' ; '+'dencity: '+str(contact_density)+' ; '+'precision: '+str(contact_precision)
                plt.text(x_label,y_label,contact_info)	
		plt.subplot(2,2,3)
          #      plt.grid(None)
                plt.imshow(figure3_matrix,cmap='binary')
		plt.scatter(flag_neg[:,1],flag_neg[:,0],s=1,c='r',marker='.')
		plt.scatter(flag_pos[:,1],flag_pos[:,0],s=1,c='g',marker='*')
		plt.plot(X,Y,c='b')
                plt.plot(Y,X,c='b')
		plt.title(title_3,y=-0.1)
		plt.subplot(2,2,4)
           #     plt.grid(None)
                plt.imshow(figure4_matrix,cmap='binary')
		plt.scatter(flag_neg[:,1],flag_neg[:,0],s=1,c='r',marker='.')
                plt.scatter(flag_pos[:,1],flag_pos[:,0],s=1,c='g',marker='*')
		plt.plot(X,Y,c='b')
                plt.plot(Y,X,c='b')
		plt.title(title_4,y=-0.1)
#		fig=plt.figure(figsize=(12,10))
#		all_data=[data_1,data_2,data_3,data_4]
#		hh=data_1.shape[0]
#		pd_data={'before':data_1,'after':data_2,'before_temp':data_3,'after_temp':data_4}
#		df=pd.DataFrame(pd_data,columns=['rmsd'])
#		title_1=name+'_dist_plot'
#		title_2=name+'_hist_plot'
#		title_3=name+'_violin_plot'
#		title_4=name+'_box_plot'
#		ax=fig.add_subplot(1,2,1)
#		ax.set_title(title_1)
#		ax.set_xlabel('point')
#		ax.set_ylabel('rmsd')
#		sns.set(style='white',font='SimHei')
#		sns.kdeplot(pd_data['before'],shade=True,color="g",label="DB_before",alpha=.7)
#		sns.kdeplot(pd_data['after'],shade=True,color="orange",label="DB_after",alpha=.7)
#		sns.kdeplot(pd_data['before_temp'],shade=True,color="blue",label="DB_before_temp",alpha=.7)
#		sns.kdeplot(pd_data['after_temp'],shade=True,color="yellow",label="DB_after_temp",alpha=.7)
#		ax=fig.add_subplot(2,2,2)
#		ax.set_title(title_2)
#		ax.set_xlabel('before')
#		ax.set_ylabel('after')
#		bins_x=np.arange(0.5,hh*0.02+0.5,0.02)
#		bins_y=np.arange(0.5,hh*0.02+0.5,0.02)
#		plt.hist2d(x=pd_data['before'],y=pd_data['after'],bins=[bins_x,bins_y])
#		plt.colorbar()
#		ax=fig.add_subplot(1,2,2)
#		ax.set_title(title_3)
#		ax.set_xlabel('point')
##		ax.set_ylabel('rmsd')
#		ax.violinplot(all_data,showmeans=True,showmedians=False)
#		ax.yaxis.grid(True)
#		plt.setp(ax,xticks=[y+1 for y in range(len(all_data))],xticklabels=['P_origin','P_after','P_origin_temp','P_after_temp'])
#		ax=fig.add_subplot(2,2,4)
#		ax.set_title(title_4)
#		ax.set_xlabel('point')
#		ax.set_ylabel('rmsd')
#		plt.boxplot(all_data)
	#	axes[0].set_title(title1)
	#	axes[1].boxplot(all_data)
	#	axes[1].set_title(title2)
	#	for ax in axes:
	#		ax.yaxis.grid(True)
	#		ax.set_xticks([y+1 for y in range(len(all_data))],)
	#		ax.set_xlabel('Filtered with Double Bubble')
	#		ax.set_ylabel('RMSD')
	#	plt.setp(axes,xticks=[y+1 for y in range(len(all_data))],xticklabels=['P_origin','P_after'])
		plt.savefig(jpgname)
		plt.close('all')


if __name__=="__main__":
	draw_image(name_list="pdb_list.txt")

