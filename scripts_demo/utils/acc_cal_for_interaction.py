#!/usr/bin/env python

import os
import numpy as np
import math

def save_distance(y_out,y,dir_name,pdb_name):
    L1 = y.shape[0]
    L2=y.shape[1]
    L=L1 +L2
    p_dict={}
    y_dict={}
    for i in range(0,L1):
        for j in range(0,L2):
            p_dict[(i,j)]=y_out[i,j]
            y_dict[(i,j)]=y[i,j]
    #top_pairs=[]
    #x=0
    x=L1*L2
    f=open(dir_name+pdb_name+'.txt','w')
    for pair in sorted(p_dict.items(), key=lambda x: x[1],reverse=True):
        (k, v) = pair
        (i,j)=k
        xx=i+1
        yy=j+1
        f.write("%d %d %.2f %.2f\n" % (xx,yy,y_dict[k],p_dict[k]))
        #top_pairs.append(k)
        #if v <15
#        x += 1
        #if v > 8:
        #    break
        #x += 1
        x-=1
        if x == 0:
            break    
    f.close()


def topKaccuracy_temp(y_out, y, k):
    L1 = y.shape[0]
    L2=y.shape[1]
    L=L1 +L2
    m = np.ones_like(y, dtype=np.int8)
#    lm = np.triu(m, 24)
 #   mm = np.triu(m, 12)
  #  sm = np.triu(m, 6)
    
   # sm = sm - mm
    #mm = mm - lm

    #avg_pred = (y_out + y_out.transpose((1, 0))) / 2.0
    truth = np.concatenate((y_out[..., np.newaxis], y[..., np.newaxis]), axis=-1)

    accs = []
#    for x in [lm, mm, sm]:
    selected_truth = truth[m.nonzero()]
    selected_truth_sorted = selected_truth[(selected_truth[:, 0]).argsort(-1)[::-1]]
    tops_num = min(selected_truth_sorted.shape[0], L//k)
    truth_in_pred = selected_truth_sorted[:, 1].astype(np.int8)
    corrects_num = np.bincount(truth_in_pred[0: tops_num], minlength=2)
    acc = 1.0 * corrects_num[1] / (tops_num + 0.0001)
    #accs.append(acc)

    return acc

def density_patch(y_out, y):
    patch_size=8
    cutoff=0.5
    L1 = y.shape[0]
    L2=y.shape[1]
    L=L1 +L2
    l1_=math.ceil(float(L1)/patch_size)
    l2_=math.ceil(float(L2)/patch_size)
    l1=int(l1_)
    l2=int(l2_)
    T_patch=np.zeros((l1,l2))
    P_patch=np.zeros((l1,l2))
    for i in range(l1):
        for j in range(l2):
            ii=i*patch_size
            iii=(i+1)*patch_size
            jj=j*patch_size
            jjj=(j+1)*patch_size
            T_patch[i,j]=np.sum(y[ii:iii,jj:jjj])
            P_patch[i,j]=np.sum(y_out[ii:iii,jj:jjj])
    T_patch[T_patch<8]=0
    T_patch[T_patch>=8]=1
    num=T_patch.sum()
    contact_density=float(num)/(l1*l2)
    return contact_density

def topK_patch_accuracy(y_out, y, k):
    patch_size=8
    cutoff=0.5
    L1 = y.shape[0]
    L2=y.shape[1]
    L=L1 +L2
    l1_=math.ceil(float(L1)/patch_size)
    l2_=math.ceil(float(L2)/patch_size)
    l1=int(l1_)
    l2=int(l2_)
    T_patch=np.zeros((l1,l2))
    P_patch=np.zeros((l1,l2))
    for i in range(l1):
        for j in range(l2):
            ii=i*patch_size
            iii=(i+1)*patch_size
            jj=j*patch_size
            jjj=(j+1)*patch_size
            T_patch[i,j]=np.sum(y[ii:iii,jj:jjj])
            P_patch[i,j]=np.sum(y_out[ii:iii,jj:jjj])
    T_patch[T_patch<8]=0
    T_patch[T_patch>=8]=1
    m = np.ones_like(T_patch, dtype=np.int8)
#    lm = np.triu(m, 24)
 #   mm = np.triu(m, 12)
  #  sm = np.triu(m, 6)
    
   # sm = sm - mm
    #mm = mm - lm

    #avg_pred = (y_out + y_out.transpose((1, 0))) / 2.0
    truth=np.concatenate((P_patch[..., np.newaxis], T_patch[..., np.newaxis]), axis=-1)
    #truth = np.concatenate((y_out[..., np.newaxis], y[..., np.newaxis]), axis=-1)

    accs = []
#    for x in [lm, mm, sm]:
    selected_truth = truth[m.nonzero()]
    selected_truth_sorted = selected_truth[(selected_truth[:, 0]).argsort(-1)[::-1]]
    tops_num = min(selected_truth_sorted.shape[0], L//k)
    truth_in_pred = selected_truth_sorted[:, 1].astype(np.int8)
    corrects_num = np.bincount(truth_in_pred[0: tops_num], minlength=2)
    acc = 1.0 * corrects_num[1] / (tops_num + 0.0001)
    #accs.append(acc)

    return acc


def topKaccuracy_temp2(y_out, y, k):
    L1 = y.shape[0]
    L2=y.shape[1]
    L=L1 +L2
    m = np.ones_like(y, dtype=np.int8)
#    lm = np.triu(m, 24)
 #   mm = np.triu(m, 12)
  #  sm = np.triu(m, 6)

   # sm = sm - mm
    #mm = mm - lm

    #avg_pred = (y_out + y_out.transpose((1, 0))) / 2.0
    truth = np.concatenate((y_out[..., np.newaxis], y[..., np.newaxis]), axis=-1)

    accs = []
#    for x in [lm, mm, sm]:
    selected_truth = truth[m.nonzero()]
    selected_truth_sorted = selected_truth[(selected_truth[:, 0]).argsort()[::-1]]
    tops_num = min(selected_truth_sorted.shape[0], L/k)
    truth_in_pred = selected_truth_sorted[:, 1].astype(np.int8)
    corrects_num = np.bincount(truth_in_pred[0: tops_num], minlength=2)
    acc = 1.0 * corrects_num[1] / (tops_num + 0.0001)
    #accs.append(acc)

    return acc

def top_1_of_K_temp(y_out, y, k):
    L1 = y.shape[0]
    L2=y.shape[1]
    L=L1*L2
    m = np.ones_like(y, dtype=np.int8)
#    lm = np.triu(m, 24)
 #   mm = np.triu(m, 12)
  #  sm = np.triu(m, 6)

   # sm = sm - mm
    #mm = mm - lm

    #avg_pred = (y_out + y_out.transpose((1, 0))) / 2.0
    truth = np.concatenate((y_out[..., np.newaxis], y[..., np.newaxis]), axis=-1)

    accs = []
#    for x in [lm, mm, sm]:
    selected_truth = truth[m.nonzero()]
    selected_truth_sorted = selected_truth[(selected_truth[:, 0]).argsort(-1)[::-1]]
    tops_num = min(selected_truth_sorted.shape[0], L/k)
    truth_in_pred = selected_truth_sorted[:, 1].astype(np.int8)
    corrects_num = np.bincount(truth_in_pred[0: tops_num], minlength=2)
    acc = 1.0 * corrects_num[1] / (tops_num + 0.0001)
    #accs.append(acc)

    return acc
def top_1_of_K_temp2(y_out, y, k):
    L1 = y.shape[0]
    L2=y.shape[1]
    L=L1*L2
    m = np.ones_like(y, dtype=np.int8)
#    lm = np.triu(m, 24)
 #   mm = np.triu(m, 12)
  #  sm = np.triu(m, 6)

   # sm = sm - mm
    #mm = mm - lm

    #avg_pred = (y_out + y_out.transpose((1, 0))) / 2.0
    truth = np.concatenate((y_out[..., np.newaxis], y[..., np.newaxis]), axis=-1)

    accs = []
#    for x in [lm, mm, sm]:
    selected_truth = truth[m.nonzero()]
    selected_truth_sorted = selected_truth[(selected_truth[:, 0]).argsort()[::-1]]
    tops_num = min(selected_truth_sorted.shape[0], L/k)
    truth_in_pred = selected_truth_sorted[:, 1].astype(np.int8)
    corrects_num = np.bincount(truth_in_pred[0: tops_num], minlength=2)
    acc = 1.0 * corrects_num[1] / (tops_num + 0.0001)
    #accs.append(acc)

    return acc

def top_mae(y_out,y):
    L1 = y.shape[0]
    L2=y.shape[1]
    L=L1+L2
 #   L=L1*L2
#    LL=L/k
    p_dict={}
    y_dict={}
    for i in range(0,L1):
        for j in range(0,L2):
            p_dict[(i,j)]=y_out[i,j]
            y_dict[(i,j)]=y[i,j]
    top_pairs=[]
    #x=0
    x=10
    for pair in sorted(p_dict.items(), key=lambda x: x[1]):
        (k, v) = pair
        top_pairs.append(k)
        #if v <15
#        x += 1
        #if v > 8:
        #    break
        #x += 1
        x-=1
        if x == 0:
            break
    sum_mae = 0.0
    sum_me=0.0
    for pair in top_pairs:
        abs_dist = abs(y_dict[pair] - p_dict[pair])
        sum_mae += abs_dist
    sum_mae /= 10
   # avg_mae += sum_mae
#        avg_mae += sum_mae
    for pair in top_pairs:
        abs_dist = y_dict[pair] - p_dict[pair]
        sum_me += abs_dist
    sum_me /= 10
   # avg_me += sum_me
#    print('MAE for ' + ' = %.2f' % sum_mae)
 #   print('ME for ' +  ' = %.2f' % sum_me)
    return sum_mae

def topK_temp(y_out, y, k):
    L = y.shape[0]

    m = np.ones_like(y, dtype=np.int8)
#    lm = np.triu(m, 24)
 #   mm = np.triu(m, 12)
  #  sm = np.triu(m, 6)

   # sm = sm - mm
    #mm = mm - lm

    #avg_pred = (y_out + y_out.transpose((1, 0))) / 2.0
    truth = np.concatenate((y_out[..., np.newaxis], y[..., np.newaxis]), axis=-1)

    accs = []
#    for x in [lm, mm, sm]:
    selected_truth = truth[m.nonzero()]
    selected_truth_sorted = selected_truth[(selected_truth[:, 0]).argsort(-1)[::-1]]
    tops_num = min(selected_truth_sorted.shape[0], k)
    truth_in_pred = selected_truth_sorted[:, 1].astype(np.int8)
    corrects_num = np.bincount(truth_in_pred[0: tops_num], minlength=2)
    acc = 1.0 * corrects_num[1] / (tops_num + 0.0001)
    #accs.append(acc)

    return acc
def topK_temp2(y_out, y, k):
    L = y.shape[0]

    m = np.ones_like(y, dtype=np.int8)
#    lm = np.triu(m, 24)
 #   mm = np.triu(m, 12)
  #  sm = np.triu(m, 6)

   # sm = sm - mm
    #mm = mm - lm

    #avg_pred = (y_out + y_out.transpose((1, 0))) / 2.0
    truth = np.concatenate((y_out[..., np.newaxis], y[..., np.newaxis]), axis=-1)

    accs = []
#    for x in [lm, mm, sm]:
    selected_truth = truth[m.nonzero()]
    selected_truth_sorted = selected_truth[(selected_truth[:, 0]).argsort()[::-1]]
    tops_num = min(selected_truth_sorted.shape[0], k)
    truth_in_pred = selected_truth_sorted[:, 1].astype(np.int8)
    corrects_num = np.bincount(truth_in_pred[0: tops_num], minlength=2)
    acc = 1.0 * corrects_num[1] / (tops_num + 0.0001)
    #accs.append(acc)

    return acc

def density(y_out, y):
    L1 = y.shape[0]
    L2=y.shape[1]
    L=L1 +L2
    num=y.sum()
    contact_density=num/L
    return contact_density

def density2(y_out, y):
    L1 = y.shape[0]
    L2=y.shape[1]
    L=L1 +L2
    y[y<8]=1
    y[y>8]=0
    y[y==8]=1
    num=y.sum()
    contact_density=num/L
    return contact_density
def evaluate_temp(predict_matrix, contact_matrix):
    #top_min_mae=top_mae(predict_matrix, contact_matrix)
   # contact_matrix[contact_matrix<8]=1
   # contact_matrix[contact_matrix==8]=1
   # contact_matrix[contact_matrix>8]=0
    #acc_k_1 = topK_patch_accuracy(predict_matrix, contact_matrix, 1)
    #acc_k_2 = topKaccuracy_temp(predict_matrix, contact_matrix, 2)
    acc_k_1 = topK_patch_accuracy(predict_matrix, contact_matrix, 1)
    acc_k_2 = topK_patch_accuracy(predict_matrix, contact_matrix, 2)
    acc_k_5 = topK_patch_accuracy(predict_matrix, contact_matrix, 5)
    acc_k_10 = topK_patch_accuracy(predict_matrix, contact_matrix, 10)
    acc_top_5=topK_temp(predict_matrix, contact_matrix, 5)
    acc_top_10=topK_temp(predict_matrix, contact_matrix, 10)
    acc_top_20=topK_temp(predict_matrix, contact_matrix, 20)
    acc_top_1_of_K=top_1_of_K_temp(predict_matrix, contact_matrix, 1000)
    #contact_density=density(predict_matrix, contact_matrix)
    contact_density=density_patch(predict_matrix, contact_matrix)
#    top_min_mae=top_mae(predict_matrix, contact_matrix)
    tmp = []
    tmp.append(acc_k_1)
    tmp.append(acc_k_2)
    tmp.append(acc_k_5)
    tmp.append(acc_k_10)
    tmp.append(acc_top_5)
    tmp.append(acc_top_10)
    tmp.append(acc_top_20)
    tmp.append(acc_top_1_of_K)
    tmp.append(contact_density)
   # tmp.append(top_min_mae)
    return tmp
def evaluate_mae(predict_matrix, contact_matrix):
    #tmp = []
    top_min_mae=top_mae(predict_matrix, contact_matrix)
    return top_min_mae

def evaluate_temp2(predict_matrix, contact_matrix):
    acc_k_1 = topKaccuracy_temp2(predict_matrix, contact_matrix, 1)
    acc_k_2 = topKaccuracy_temp2(predict_matrix, contact_matrix, 2)
    acc_k_5 = topKaccuracy_temp2(predict_matrix, contact_matrix, 5)
    acc_k_10 = topKaccuracy_temp2(predict_matrix, contact_matrix, 10)
    acc_top_5=topK_temp2(predict_matrix, contact_matrix, 5)
    acc_top_10=topK_temp2(predict_matrix, contact_matrix, 10)
    acc_top_20=topK_temp2(predict_matrix, contact_matrix, 20)
    acc_top_1_of_K=top_1_of_K_temp2(predict_matrix, contact_matrix, 1000)
    contact_density=density2(predict_matrix, contact_matrix)
    tmp = []
    tmp.append(acc_k_1)
    tmp.append(acc_k_2)
    tmp.append(acc_k_5)
    tmp.append(acc_k_10)
    tmp.append(acc_top_5)
    tmp.append(acc_top_10)
    tmp.append(acc_top_20)
    tmp.append(acc_top_1_of_K)
    tmp.append(contact_density)
    return tmp
def output_result_temp(avg_acc,mae):
#    print "Acc :     %.3f" %acc_k_1
    print "toplen    L/10         L/5          L/2        L        top5        top10        top20        top_1_of_1000        contact_density    mae"
    print "Acc :     %.3f         %.3f         %.3f       %.3f       %.3f       %.3f       %.3f       %.3f       %.3f       %.3f" \
            %(avg_acc[3], avg_acc[2], avg_acc[1], avg_acc[0],avg_acc[4],avg_acc[5],avg_acc[6],avg_acc[7],avg_acc[8],mae)

def output_result_number(avg_acc):
    num=np.random.random(9)
    num_matrix=(avg_acc>0)+0
    for i in range(9):
#    	temp=(avg_acc[:,i]>0)
	num[i]=sum(num_matrix[:,i])

#    print "Acc :     %.3f" %acc_k_1
    print "toplen    L/10         L/5          L/2        L        top5        top10        top20        top_1_of_1000        contact_density"
    print "Acc :     %.3f         %.3f         %.3f       %.3f       %.3f       %.3f       %.3f       %.3f       %.3f" \
            %(num[3], num[2], num[1], num[0],num[4],num[5],num[6],num[7],num[8])

def output_result(avg_acc):
    print "Long Range(> 24):"
    print "Method    L/10         L/5          L/2        L"
    print "Acc :     %.3f        %.3f        %.3f      %.3f" \
            %(avg_acc[3][0], avg_acc[2][0], avg_acc[1][0], avg_acc[0][0])
    print "Medium Range(12 - 24):"
    print "Method    L/10         L/5          L/2        L"
    print "Acc :     %.3f        %.3f        %.3f      %.3f" \
            %(avg_acc[3][1], avg_acc[2][1], avg_acc[1][1], avg_acc[0][1])
    print "Short Range(6 - 12):"
    print "Method    L/10         L/5          L/2        L"
    print "Acc :     %.3f        %.3f        %.3f      %.3f" \
            %(avg_acc[3][2], avg_acc[2][2], avg_acc[1][2], avg_acc[0][2])

def test():
    with open("data/PSICOV/psicov.list", "r") as fin:
        names = [line.rstrip("\n") for line in fin]

    accs = []
    for i in range(len(names)):
        name = names[i]
        print "processing in %d: %s" %(i+1, name)
        
        #prediction_path = "data/PSICOV/clm/"
        prediction_path = "data/PSICOV/new_psicov/"
        #prediction_path = "data/PSICOV/psicov_matrix"
        #prediction_path = "data/PSICOV/mf_matrix"
        #prediction_path = "psicov_result"
        f = os.path.join(prediction_path, name + ".ccmpred")
        if not os.path.exists(f):
            print "not exist..."
            continue
        y_out = np.loadtxt(f)

        dist_path = "data/PSICOV/dis/"
        y = np.loadtxt(os.path.join(dist_path, name + ".dis"))
        y[y > 8] = 0 
        y[y != 0] = 1
        y = y.astype(np.int8)
        y = np.tril(y, k=-6) + np.triu(y, k=6) 

        acc = evaluate(y_out, y)
        accs.append(acc)
    accs = np.array(accs)
    avg_acc = np.mean(accs, axis=0)
    output_result(avg_acc)

