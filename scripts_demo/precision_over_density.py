# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:02:23 2019

@author: huangh
"""

from __future__ import division
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import precision_recall_curve
from scipy import interp
import os
import random
import matplotlib
import numpy as np
matplotlib.use('agg')
import matplotlib.pyplot as plt
#计算样本中正例和负例的样本数
def read_data(file_path):
    #[score,bool,bool] 输出概率和其真实标签，如果为为正则第二列为1如果为负样本则第三个为1否则为0
    name_temp=file_path.split('.')
    name=name_temp[0]
    samples = []
    pos, neg = 0, 0 #真实的正负样本数
    aa=np.loadtxt(file_path)
    L=aa.shape[0]
#    bb=np.zeros((L,3)
    for i in range(L):
        score=aa[i,3]
        score_true=aa[i,2]
        temple=[score, 1, 0] if score_true >0 else [score, 0, 1]
        pos += temple[1]
        neg += temple[2]
        samples.append(temple)
    return samples, pos , neg,name
        

#    with open(file_path,'r') as f:
 #       for line in f:
  #          temp = line.split("\t")
   #         #score = '%.2f'% float(temp[0])
     #       score = float((temp[2]+temp[3])/2)
    #        score_true=float((temp[7]+temp[8])/2)
#            true_label = score
      #      temple = [score, 1, 0] if score_true < 8.0 else [score, 0, 1]
       #     pos += temple[1]
        #    neg += temple[2]
         #   samples.append(temple)
#    return samples, pos , neg,name

#输出概率从大到小排序，并计算假阳率和真阳率
def sort_roc(samples, pos, neg):
    fp, tp = 0, 0 #假阳，真阳
    xy_fpr_tpr = []
    xy_recall_precision=[]
    sample_sort = sorted(samples, key = lambda x:x[0])
#    file=open('data.txt','w')
#    file.write(str(sample_sort))
#    file.close()
    for i in range(len(sample_sort)):
        fp += sample_sort[i][2]
        tp += sample_sort[i][1]
	tn =neg-fp
	fn =pos -tp
	TPR_sum=tp + fn
	FPR_sum=fp +tn
	tpfp=tp+fp 	
        xy_fpr_tpr.append([fp/FPR_sum, tp/TPR_sum])
	xy_recall_precision.append([tp/TPR_sum,tp/tpfp])
#    file.write(str(xy_fpr_tpr))
#    file.close()
    return xy_fpr_tpr,xy_recall_precision
#画出ROC
def get_auc(xy_fpr_tpr):
    auc = 0.0
    pre_x = 0
    for x,y in xy_fpr_tpr:
        if x != pre_x:
            auc += (x-pre_x)*y
            pre_x = x
    return auc

def draw_roc(xy_fpr_tpr,name):
    x = [item[0] for item in xy_fpr_tpr]
    y = [item[1] for item in xy_fpr_tpr]
    image_name=os.path.join('./plot_image/',name+'_roc'+'.png')
    plt.plot(x,y)
    plt.title('ROC curve (AUC = %.4f)' % get_auc(xy_fpr_tpr))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(image_name)
    plt.close()

def draw_line(xy_fpr_tpr,name):
    x = [item[0] for item in xy_fpr_tpr]
    y = [item[1] for item in xy_fpr_tpr]
    image_name=os.path.join('./plot_image/',name+'_precision'+'.png')
    plt.plot(x,y)
    plt.title('precision curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(image_name)
    plt.close()
            
if __name__ == '__main__':
   # M=np.random.random((1500,150))
   # N=np.random.random((1500,150))
   # M_temp=np.random.random((1500,150))
   # N_temp=np.random.random((1500,150))
   # i=0
    tprs=[]
    aucs=[]
    tprs_=[]
    aucs_=[]
    mean_fpr=np.linspace(0,1,100)
    mean_fpr_=np.linspace(0,1,100)
    xy_temp=[]
    XY_temp=[]
    zz=[]
    plt.figure()
   # plt.subplot(1,2,1)
    i=0
    num=10
    precision=[]
    precision_r=[]
    density=[]
    for filename in os.listdir('./masif_homo_predictions_rr_all'):
      #  temp=filename.split('.')
      
        file_path=os.path.join('./masif_homo_predictions_rr',filename)
        a=np.loadtxt(file_path)
        pred=a[:num,3]
        true=a[:num,2]    
        length=true.shape[0]
        random_list=range(0,length)
        random_pred=random.sample(random_list,length)
        ss=np.random.random((length,2))
        ss[:,0]=true
        ss[:,1]=random_pred
        sss=ss[np.argsort(-ss[:,1]),:]
        true_r=sss[:num,0]
        pred_r=sss[:num,1]
        
       # fpr_,tpr_,thresholds_=roc_curve(true_r,pred_r)
       # fpr,tpr,thresholds=roc_curve(true,pred)
     #   pr_,re_,thresholds_=precision_recall_curve(true_r,pred_r)
      #  pr,re,thresholds=precision_recall_curve(true,pred)
        pr=np.sum(a[:num,2])/num
        den=np.sum(a[:,2])/(np.sqrt(a.shape[0]))
        precision.append(pr)
   #     precision_r.append(pr_[-1])
        density.append(den)
      #  tprs.append(interp(mean_fpr,fpr,tpr))
      #  tprs[-1][0]=0.0
      #  roc_auc=auc(fpr,tpr)
      #  aucs.append(roc_auc)
      #  tprs_.append(interp(mean_fpr_,fpr_,tpr_))
      #  tprs_[-1][0]=0.0
      #  roc_auc_=auc(fpr_,tpr_)
      #  aucs_.append(roc_auc_)
      #  plt.plot(fpr,tpr,lw=1,alpha=0.3)
#        plt.plot(fpr,tpr,lw=1,alpha=0.3)
        i=i+1
     #   if i >2:
      #      break
    
    hh_=np.concatenate([np.reshape(density,(i,1)),np.reshape(precision,(i,1))],-1)
    hh=np.concatenate([hh_,np.reshape(precision,(i,1))],-1)
    hhhh=hh[np.argsort(hh[:,0]),:]
    min_=hhhh[0,0]
    max_=hhhh[-1,0]
    val=(max_-min_)/10.0
    p=[]
    p_r=[]
    d=[]
    for j in range(0,10):
        k1=j*val+min_
        k2=(j+1)*val+min_
        n1=hhhh[hhhh[:,0]<=k2].shape[0]
        n2=hhhh[hhhh[:,0]<k1].shape[0]
        n3=n1-n2
#	print(n3)
        p1=np.sum(hhhh[n2:n1,1])/n3
        p2=np.sum(hhhh[n2:n1,2])/n3
        dd=(k1+k2)/2.0
        p.append(p1)
        p_r.append(p2)
        d.append(dd)
#    plt.bar(d,p)
    dddd_=np.array(d)
    dddd=np.reshape(dddd_,(10,1))
    pppp_=np.array(p)
    pppp=np.reshape(pppp_,(10,1))
    plt.plot(dddd,pppp,color='b')
  #  plt.scatter(dddd,pppp,marker='o',s=10)
#    for i in range(0,10):
 #       plt.text(dddd[i],pppp[i],'%.0f'%pppp[i])
#    plt.plot([1,0],[0,1],linestyle='--',lw=2,color='r',label='Luck',alpha=.8)
#    mean_tpr=np.mean(tprs,axis=0)
 #   mean_tpr[-1]=0
  #  mean_auc=auc(mean_fpr,mean_tpr)
   # std_auc=np.std(tprs,axis=0)
   # plt.plot(mean_fpr,mean_tpr,color='b',label=r'Mean ROC (area=%0.2f)'%mean_auc,lw=2,alpha=.8)
   # mean_tpr_=np.mean(tprs_,axis=0)
   # mean_tpr_[-1]=0
   # mean_auc_=auc(mean_fpr_,mean_tpr_)
   # std_auc_=np.std(tprs_,axis=0)
   # plt.plot(mean_fpr_,mean_tpr_,color='y',label=r'Random ROC (area=%0.2f)'%mean_auc_,lw=2,alpha=.8)
   # std_tpr=np.std(tprs,axis=0)
   # tprs_upper=np.minimum(mean_tpr+std_tpr,1)
   # tprs_lower=np.maximum(mean_tpr-std_tpr,0)
   # plt.fill_between(mean_tpr,tprs_lower,tprs_upper,color='gray',alpha=.2)
    plt.xlim([-0.05,10.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('density')
    plt.ylabel('Accuracy')
    plt.title('Acc')
    plt.legend(loc='lower right')
    plt.savefig("Acc_density_homo_0121.jpg") 

#        samples, pos, neg,name = read_data(file_path)
 #       xy,XY = sort_roc(samples, pos, neg)
  #      x = [item[0] for item in xy]
   #     y = [item[1] for item in xy]
    #    z=get_auc(xy)
     #   plt.plot(x,y,lw=1,alpha=0.1)
      #  zz.append(z)
       # xy_temp.append(xy)
       # XY_temp.append(XY)
    # plt.plot(x,y,lw=1,alpha=0.1)
  #      xy_1=[item[0] for item in xy]
   #     xy_2=[item[1] for item in xy]
    #    XY_1=[item[0] for item in XY]
     #   XY_2=[item[1] for item in XY]
      #  a=np.array(XY_1)
       # ll=a.shape[0]
       # if ll > 1500:
        #    ll=1500
#        M[:ll,i]=XY_1[0:ll]
 #       N[:ll,i]=XY_2[0:ll]
  #      M_temp[:ll,i]=xy_1[:ll]
   #     N_temp[:ll,i]=xy_2[:ll]
    #    i+=1
  #  name='AUC_01_19'
#    M1=np.mean(M,axis=1)
 #   M2=np.mean(N,axis=1)
  #  M3=np.mean(M_temp,axis=1)
   # M4=np.mean(N_temp,axis=1)
   # XY=np.random.random((1500,2))
   # xy=np.random.random((1500,2))
   # XY[:,0]=M1
   # XY[:,1]=M2
   # xy[:,0]=M3
   # xy[:,1]=M4
   # draw_line(XY,name)
   # draw_roc(xy,name)

        
   
