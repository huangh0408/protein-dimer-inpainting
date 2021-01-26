import datetime
import ipdb
import os
from glob import glob
import pandas as pd
import numpy as np
import cv2
from model import *
from util import *
import scipy.misc as misc
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
#import keras as K
import sys
import argparse
config=tf.ConfigProto()
#config=tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.8
session=tf.Session(config=config)
#session=tf.compat.v1.Session(config=config)

wirter = tf.summary.FileWriter('logs/',session.graph)

#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

parser = argparse.ArgumentParser()
parser.add_argument("--Gpuid",type=int,default=1, help="choose the  gpu")
parser.add_argument("--input_dir",default="./datasets", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["contact", "distance", "slice"])
parser.add_argument("--type", required=True, choices=["homodimer", "heteromer", "all"])
parser.add_argument("--datafile",default="./data", help="path to folder containing pickle files")
parser.add_argument("--netsize", type=int,required=True, choices=[128, 256, 512])
parser.add_argument("--output_dir", default="./results", help="where to put output files")
parser.add_argument("--checkpoint", default="./models", help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--pretrained_dir", default=None, help="directory with pretrained models")
parser.add_argument("--max_epochs", type=int, default=500,help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=100, help="number of images in batch")

hh = parser.parse_args()

#os.environ["CUDA_VISIBLE_DEVICES"]=hh.Gpuid
input_name=hh.input_dir
dataset_type=hh.mode
dataset_source=hh.type
data_pickle=hh.datafile
net_size=hh.netsize
output_name=hh.output_dir
model_name=hh.checkpoint
pretrained_name=hh.pretrained_dir
n_epochs = hh.max_epochs
batch_size = hh.batch_size

dataset_path=os.path.join(input_name,dataset_type)
result_path= os.path.join(output_name,dataset_source+'_'+dataset_type+'_'+str(net_size)+'/')
model_path = os.path.join(model_name,dataset_source+'_'+dataset_type+'_'+str(net_size)+'/')
trainset_path = os.path.join(data_pickle,'protein_trainset_'+dataset_source+'_'+dataset_type+'_size'+str(net_size)+'.pickle')
testset_path  = os.path.join(data_pickle,'protein_valset_'+dataset_source+'_'+dataset_type+'_size'+str(net_size)+'.pickle')
#pretrained_model_path =pretrained_name
if pretrained_name is not None:
    pretrained_model_path =None
pretrained_model_path =None

#learning_rate_val = 0.005
#weight_decay_rate =  0.0001
learning_rate_val = 0.0025
weight_decay_rate =  0.0001
momentum = 0.9
lambda_recon = 0.9
lambda_adv = 0.1
#beta_local=0.8
#beta_global=0.2
overlap_size = 0
hiding_size = net_size/2


if not os.path.exists(model_path):
    os.makedirs( model_path )

if not os.path.exists(result_path):
    os.makedirs( result_path )

if not os.path.exists( trainset_path ) or not os.path.exists( testset_path ):

    trainset_dir = os.path.join( dataset_path, 'train_images' )
    trainset_dir_temp=os.listdir(trainset_dir)
    trainset_dir_temp.sort()
    testset_dir = os.path.join( dataset_path, 'eval_images' )
    testset_dir_temp=os.listdir(testset_dir)
    testset_dir_temp.sort()
    trainset_chain_dir = os.path.join( dataset_path, 'train_chains' )
    trainset_chain_dir_temp=os.listdir(trainset_chain_dir)
    trainset_chain_dir_temp.sort()
    testset_chain_dir = os.path.join( dataset_path, 'eval_chains' )
    testset_chain_dir_temp=os.listdir(testset_chain_dir)
    testset_chain_dir_temp.sort()
    trainset = pd.DataFrame({'image_path': map(lambda x: os.path.join( trainset_dir, x ), trainset_dir_temp),'chain_path':map(lambda x: os.path.join( trainset_chain_dir, x ), trainset_chain_dir_temp)})
    testset = pd.DataFrame({'image_path': map(lambda x: os.path.join( testset_dir, x ), testset_dir_temp),'chain_path':map(lambda x: os.path.join( testset_chain_dir, x ), testset_chain_dir_temp)})

    trainset.to_pickle( trainset_path )
    testset.to_pickle( testset_path )
else:
    trainset = pd.read_pickle( trainset_path )
    testset = pd.read_pickle( testset_path )

#testset.index = range(len(testset))
#testset = testset.ix[np.random.permutation(len(testset))]
is_train = tf.placeholder( tf.bool )

learning_rate = tf.placeholder( tf.float32, [])
ll=tf.placeholder( tf.int32, [batch_size])
net_size_tf=tf.placeholder( tf.int32, [net_size])
images_tf = tf.placeholder( tf.float32, [batch_size, net_size, net_size, 3], name="images")
#images_global = tf.placeholder( tf.float32, [batch_size, net_size, net_size, 3], name="images_global")
#reconstruction_global=tf.placeholder( tf.float32, [batch_size, 128, 128, 3], name="rec_global")
fake_length=5
true_length=batch_size-fake_length
label_l=2*batch_size
labels_D = tf.concat( [tf.ones([batch_size]), tf.zeros([batch_size])] ,0)
#labels_D=tf.placeholder( tf.float32, [label_l])
#labels_D = tf.concat( [tf.random_uniform([batch_size],minval=0,maxval=0.2,dtype=tf.float32), tf.random_uniform([batch_size],minval=0.8,maxval=1.0,dtype=tf.float32)] ,0)
#labels_D = tf.concat( [tf.ones([true_length]),tf.zeros([fake_length]), tf.ones([fake_length]),tf.zeros([true_length])] ,0)
#labels_D_global = tf.concat( [tf.random_uniform([batch_size],minval=0,maxval=0.2,dtype=tf.float32), tf.random_uniform([batch_size],minval=0.8,maxval=1.0,dtype=tf.float32)] ,0)
#labels_D_global = tf.concat( [tf.ones([true_length]),tf.zeros([fake_length]), tf.ones([fake_length]),tf.zeros([true_length])] ,0)
labels_G = tf.ones([batch_size])
#images_hiding = tf.placeholder( tf.float32, [batch_size, hiding_size, hiding_size, 3], name='images_hiding')
images_hiding = tf.placeholder( tf.float32, [batch_size, net_size, net_size, 3], name='images_hiding')

model = Model()

#bn1, bn2, bn3, bn4, bn5, bn6, debn4, debn3, debn2, debn1, reconstruction_ori, reconstruction,reconstruction_global = model.build_reconstruction(images_tf,images_global,ll, is_train)
bn1, bn2, bn3, bn4, bn5, bn6, debn4, debn3, debn2, debn1, reconstruction_ori, reconstruction = model.build_reconstruction(images_tf,ll,net_size, is_train)
#for i in range(batch_size):
#    recon_reshape=tf.image.resize_images(reconstruction[i,:,:,:],[ll[i],net_size-ll[i]])
#    names=locals()
 #   if i = 0:
#        names['rec_5_'+str(0)]=
#    names['rec_global_'+str(i)]=tf.reshape(recon_reshape,[ll[i],net_size-ll[i],3])
#    names['rec_1_'+str(i)]=images_tf[i,0:ll[i],0:ll[i],:]
#    names['rec_2_'+str(i)]=images_tf[i,ll[i]:net_size,:,:]
#    names['rec_3_'+str(i)]=tf.concat([names['rec_1_'+str(i)],names['rec_global_'+str(i)]],1)
#    names['rec_4_temp_'+str(i)]=tf.concat([names['rec_3_'+str(i)],names['rec_2_'+str(i)]],0)
#    names['rec_4_'+str(i)]=tf.reshape(names['rec_4_temp_'+str(i)],[1,net_size,net_size,3])
#    if i == 0:
#        names['rec_5_'+str(0)]=names['rec_4_'+str(0)]
#    if i > 0:
#        names['rec_5_'+str(i)]=tf.concat([names['rec_5_'+str(i-1)],names['rec_4_'+str(i)]],0)
#reconstruction_global=names['rec_5_'+str(batch_size-1)]
mask_recon = tf.placeholder( tf.float32, [batch_size, net_size, net_size, 3], name='images_mask')

adversarial_pos = model.build_adversarial_temp(images_hiding*mask_recon,net_size, is_train)
adversarial_neg = model.build_adversarial_temp(reconstruction*mask_recon, net_size,is_train, reuse=True)
adversarial_all = tf.concat([adversarial_pos, adversarial_neg],0)
#images_global=tf.placeholder( tf.float32, [batch_size, 128, 128, 3], name="images_global")
#images_global_0=images_tf[:,:,:,0]
#images_global_1=images_tf[:,:,:,1]
#images_global_2=images_tf[:,:,:,2]
#im_temp_0=misc.imresize(images_hiding[:,:,:,0],[batch_size,ll,128-ll,1],interp='nearest')
#im_temp_1=misc.imresize(images_hiding[:,:,:,1],[batch_size,ll,128-ll,1],interp='nearest')
#im_temp_2=misc.imresize(images_hiding[:,:,:,2],[batch_size,ll,128-ll,1],interp='nearest')
#images_global_0[:,0:ll,ll:128,0].assign(im_temp_0)
#images_global_1[:,0:ll,ll:128,1].assign(im_temp_1)
#images_global_2[:,0:ll,ll:128,2].assign(im_temp_2)
#images_global=tf.stack([images_global_0,images_global_1,images_global_2],3)
#reconstruction_global=tf.placeholder( tf.float32, [batch_size, 128, 128, 3], name="images_hiding_global")
#rec_global_0=reconstruction[:,:,:,0]
#rec_global_1=reconstruction[:,:,:,1]
#rec_global_2=reconstruction[:,:,:,2]
#rec_temp_0=misc.imresize(reconstruction[:,:,:,0],[batch_size,ll,128-ll,1],interp='nearest')
#rec_temp_1=misc.imresize(reconstruction[:,:,:,1],[batch_size,ll,128-ll,1],interp='nearest')
#rec_temp_2=misc.imresize(reconstruction[:,:,:,2],[batch_size,ll,128-ll,1],interp='nearest')
#rec_global_0[:,0:ll,ll:128,0].assign(rec_temp_0)
#rec_global_1[:,0:ll,ll:128,1].assign(rec_temp_1)
#rec_global_2[:,0:ll,ll:128,2].assign(rec_temp_2)
#reconstruction_global=tf.stack([rec_global_0,rec_global_1,rec_global_2],3)
#reconstruction_global[:,0:ll,ll:128,:]=reconstruction
#ii = 0
#for rec_val, img,x,y in zip(reconstruction_ori, images_global, ll, ll):
#for ii in range(batch_size):
  #  rec_val1=reconstruction_ori[ii,:,:,:]
 #   img1=images_global[ii,:,:,:]
   # x=ll[ii]
   # y=x
#    sess=tf.Session()
  #  rec_val=rec_val1.eval(session=session)
  #  img=img1.eval(session=sess)
  #  rec_hid = rec_val.reshape((64,64,3))
  #  rec_con = img.reshape((128,128,3))
  #  rec_1=rec_con.copy()
 #   xx=128-x
  #  rec_2=misc.imresize(rec_hid[:,:,0],[y,xx],interp='nearest')
  #  rec_3=misc.imresize(rec_hid[:,:,1],[y,xx],interp='nearest')
  #  rec_4=misc.imresize(rec_hid[:,:,2],[y,xx],interp='nearest')
  #  rec_hid_temp=rec_1[0:y,x:128]
  #  rec_hid_temp[:,:,0]=rec_2
  #  rec_hid_temp[:,:,1]=rec_3
  #  rec_hid_temp[:,:,2]=rec_4
  #  rec_con[0:y, x:128] = rec_hid_temp
  #  bb=re_con.reshape((1,128,128,3))
  #  reconstruction_global[ii,:,:,:]=bb
#    cv2.imwrite( os.path.join(result_path, 'img_'+str(ii)+'.'+str(int(iters/100))+'.jpg'), rec_con)
#    ii += 1
#reconstruction_global=torch.tensor(reconstruction_global)
#adversarial_pos_global=model.build_adversarial_temp(images_global,net_size, is_train)
#adversarial_neg_global=model.build_adversarial_temp(reconstruction_global, net_size,is_train, reuse=True)
#adversarial_all_global = tf.concat([adversarial_pos_global, adversarial_neg_global],0)

# Applying bigger loss for overlapping region
#mask_recon = tf.placeholder( tf.float32, [batch_size, net_size, net_size, 3], name='images_mask')
#mask_recon = tf.pad(tf.ones([net_size - 2*overlap_size, net_size - 2*overlap_size]), [[overlap_size,overlap_size], [overlap_size,overlap_size]])
#mask_recon = tf.reshape(mask_recon, [net_size, net_size, 1])
#mask_recon = tf.concat([mask_recon]*3,2)
mask_overlap = 1 - mask_recon

###focal_loss###
#def focal_loss(gamma=2, alpha=0.75):
def focal_loss_fixed(y_true, y_pred):#with tensorflow
    gamma=1.3
    alpha=0.30
    eps = 1e-12
    y_pred=tf.clip_by_value(y_pred,eps,1.-eps)#improve the stability of the focal loss and see issues 1 for more information
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.log(pt_1))-tf.reduce_sum((1-alpha) * tf.pow( pt_0, gamma) * tf.log(1. - pt_0))
#    return focal_loss_fixed

#loss_recon_ori=focal_loss_fixed(images_hiding,reconstruction)
#images_hiding_one=tf.ones_like(images_hiding)
#images_hiding_zero=tf.zeros_like(images_hiding)
#images_hiding_revalue=tf.where(images_hiding>0.2,images_hiding_one,images_hiding_zero)
#loss_recon_ori = tf.abs((tf.multiply(tf.square( images_hiding - reconstruction),images_hiding_revalue)))
loss_recon_ori = tf.square( images_hiding - reconstruction )
loss_recon_center = tf.reduce_mean(tf.sqrt( 1e-5 + tf.reduce_sum(loss_recon_ori * mask_recon, [1,2,3]))) *10.0 # Loss for non-overlapping region
loss_recon_overlap = tf.reduce_mean(tf.sqrt( 1e-5 + tf.reduce_sum(loss_recon_ori * mask_overlap, [1,2,3])))  # Loss for overlapping region
loss_recon = loss_recon_center + loss_recon_overlap
#loss_recon=loss_recon_ori

#loss_adv_D = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=adversarial_all, logits=labels_D))
#loss_adv_G = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=adversarial_neg, logits=labels_G))
#loss_adv_D_global=tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=adversarial_all_global, logits=labels_D_global))
loss_adv_D = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_D, logits=adversarial_all))
loss_adv_G = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_G, logits=adversarial_neg))
#loss_adv_D_global=tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_D_global, logits=adversarial_all_global))
loss_G = loss_adv_G * lambda_adv + loss_recon * lambda_recon
loss_D = loss_adv_D # * lambda_adv

#var_G = filter( lambda x: x.name.startswith('GEN'), tf.trainable_variables())
var_G = list(filter( lambda x: x.name.startswith('GEN'), tf.trainable_variables()))
var_D = list(filter( lambda x: x.name.startswith('DIS'), tf.trainable_variables()))

W_G = list(filter(lambda x: x.name.endswith('W:0'), var_G))
W_D = list(filter(lambda x: x.name.endswith('W:0'), var_D))

#loss_G += weight_decay_rate * tf.reduce_mean(tf.stack( map(lambda x: tf.nn.l2_loss(x), W_G)))
loss_G += weight_decay_rate * tf.reduce_mean(tf.stack( list( map(lambda x: tf.nn.l2_loss(x), W_G))))

#loss_D += weight_decay_rate * tf.reduce_mean(tf.stack( map(lambda x: tf.nn.l2_loss(x), W_D)))
loss_D += weight_decay_rate * tf.reduce_mean(tf.stack( list( map(lambda x: tf.nn.l2_loss(x), W_D))))

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

optimizer_G = tf.train.AdamOptimizer( learning_rate=learning_rate, beta1=0.5 )
grads_vars_G = optimizer_G.compute_gradients( loss_G, var_list=var_G )
#grads_vars_G = map(lambda gv: [tf.clip_by_value(gv[0], -10., 10.), gv[1]], grads_vars_G)
train_op_G = optimizer_G.apply_gradients( grads_vars_G )

#optimizer_D = tf.train.AdamOptimizer( learning_rate=learning_rate, beta1=0.5 )
optimizer_D = tf.train.GradientDescentOptimizer( learning_rate=learning_rate)
grads_vars_D = optimizer_D.compute_gradients( loss_D, var_list=var_D )
#grads_vars_D = map(lambda gv: [tf.clip_by_value(gv[0], -10., 10.), gv[1]], grads_vars_D)
train_op_D = optimizer_D.apply_gradients( grads_vars_D )

saver = tf.train.Saver(max_to_keep=5)

tf.initialize_all_variables().run()

if pretrained_model_path is not None and os.path.exists( pretrained_model_path ):
    saver.restore( sess, pretrained_model_path )

iters = 0

loss_D_val = 0.
loss_G_val = 0.


log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


for epoch in range(n_epochs):
    trainset.index = range(len(trainset))
    #trainset = trainset.ix[np.random.permutation(len(trainset))]
    trainset = trainset.iloc[np.random.permutation(len(trainset))]

    if epoch < 5:
#        ss=(epoch/(5*n_epochs+0.1))
        ss=0.05
        sss=0.8+ss
        ssss=0.2-ss
    #    labels_D_hh = tf.concat( [tf.random_uniform([batch_size],minval=0.799+s,maxval=1.0,dtype=tf.float32), tf.random_uniform([batch_size],minval=0,maxval=0.201-s,dtype=tf.float32)] ,0)
        labels_D_hh = np.concatenate( (np.random.random(batch_size)*ssss, np.random.random(batch_size)*ssss+sss) ,0)
    elif epoch % 20 == 0: 
        ss=0.1
        sss=0.8+ss
        ssss=0.2-ss
        fake_length=5
        true_length=batch_size-fake_length
        labels_D_hh = np.concatenate( (np.random.random(fake_length)*ssss,np.random.random(true_length)*ssss+sss ,np.random.random(fake_length)*ssss+sss ,np.random.random(true_length)*ssss) ,0) 
        
    else:
#        ss=(epoch/(5*n_epochs+0.1))
        t=epoch/n_epochs
        ss=0.1+0.1*t
        sss=0.8+ss
        ssss=0.2-ss
#    labels_D_hh = tf.concat( [tf.random_uniform([batch_size],minval=0.799+s,maxval=1.0,dtype=tf.float32), tf.random_uniform([batch_size],minval=0,maxval=0.201-s,dtype=tf.float32)] ,0)
        labels_D_hh = np.concatenate( (np.random.random(batch_size)*ssss+sss, np.random.random(batch_size)*ssss) ,0)


    for start,end in zip(
            range(0, len(trainset), batch_size),
            range(batch_size, len(trainset), batch_size)):

        image_paths = trainset[start:end]['image_path'].values
        chain_paths = trainset[start:end]['chain_path'].values


        chain_ori= list(map(lambda x: load_chain( x,net_size ), chain_paths))
        images_ori = list(map(lambda x: load_image( x,net_size ), image_paths))

        #print("---- test here 1 ----")
        #print(list(chain_ori))
        #print(list(images_ori))

#        if iters % 2 == 0:
 #           crop_interaction_rotate
#        if iters % 2 == 0:
 #           images_ori = map(lambda img: img[:,::-1,:], images_ori)
#        if iters % 2 == 0:
            
        is_none = np.sum(list(map(lambda x: x is None, images_ori)))
        if is_none > 0:
            print("!!! is none !!!")
            continue
        else:
            print(is_none)

#        images_crops = map(lambda x: crop_random(x, x=32, y=32), images_ori)
#        images, crops,_,_ = zip(*images_crops)
#        if epoch % 2 ==0:


        if epoch % 4 ==0:
        
            images_crops = map(lambda x,y: crop_interaction_swap(x, y,net_size), images_ori,chain_ori)
            images,images_in, crops,x_t,y_t= zip(*images_crops)
        elif epoch % 3 ==0:
            images_crops = map(lambda x,y: crop_interaction_rotate(x, y,net_size), images_ori,chain_ori)
            images,images_in, crops,x_t,y_t= zip(*images_crops)
        else:
            images_crops = map(lambda x,y: crop_interaction(x, y,net_size), images_ori,chain_ori)
            images,images_in, crops,x_t,y_t= zip(*images_crops)
        
        #images_crops = map(lambda x,y: crop_interaction(x, y,net_size), images_ori,chain_ori)
        #print("---- test here 2 ------")
        #images,images_in,crops,x_t,y_t = zip(*images_crops)


        # Printing activations every 10 iterations
        if iters % 500 == 0:
            test_image_paths = testset[:batch_size]['image_path'].values
            test_chain_paths = testset[:batch_size]['chain_path'].values


            test_chain_ori=map(lambda x: load_chain( x,net_size ), test_chain_paths)
            test_images_ori = map(lambda x: load_image(x,net_size), test_image_paths)

            test_images_crop = map(lambda x,y: crop_interaction(x, y,net_size), test_images_ori,test_chain_ori)
            test_images, test_images_in,test_crops,xs,ys= zip(*test_images_crop)

            reconstruction_vals, recon_ori_vals,bn1_val,bn2_val,bn3_val,bn4_val,bn5_val,bn6_val,debn4_val, debn3_val, debn2_val, debn1_val, loss_G_val, loss_D_val = sess.run(
                    [reconstruction, reconstruction_ori, bn1,bn2,bn3,bn4,bn5,bn6,debn4, debn3, debn2, debn1, loss_G, loss_D],
                    feed_dict={
                        images_tf: test_images,
                        mask_recon:test_crops,
                        images_hiding: test_images_in,
                        ll:xs,
#                        net_size:net_size_tf,
                        is_train: False
                        })

            # Generate result every 1000 iterations
            if iters % 500 == 0:
                ii = 0
                for rec_val, img,x,y in zip(reconstruction_vals, test_images, xs, ys):
                    rec_hid = (255. * rec_val).astype(int)
                    rec_con = (255. * img).astype(int)
                    rec_1=rec_con.copy()
                    xx=net_size-x
#                    rec_2=misc.imresize(rec_hid[:,:,0],[y,xx],interp='nearest')
 #                   rec_3=misc.imresize(rec_hid[:,:,1],[y,xx],interp='nearest')
  #                  rec_4=misc.imresize(rec_hid[:,:,2],[y,xx],interp='nearest')
                    rec_hid_temp=rec_1[0:y,x:net_size]
                    rec_hid_temp[:,:,0]=rec_hid[0:y, x:net_size,0]
                    rec_hid_temp[:,:,1]=rec_hid[0:y, x:net_size,1]
                    rec_hid_temp[:,:,2]=rec_hid[0:y, x:net_size,2]
                    rec_con[0:y, x:net_size] = rec_hid_temp
                    cv2.imwrite( os.path.join(result_path, 'img_'+str(ii)+'.'+str(int(iters/100))+'.jpg'), rec_con)
                    ii += 1

                if iters == 0:
                    ii=0
                    for test_image in test_images_ori:
                        test_image = (255. * test_image).astype(int)
                        cv2.imwrite( os.path.join(result_path, 'img_'+str(ii)+'.true.jpg'), test_image)
                        test_image[0:y, x:net_size] = 0
                        cv2.imwrite( os.path.join(result_path, 'img_'+str(ii)+'.ori.jpg'), test_image)
                        ii+=1

            print("========================================================================")
            print(bn1_val.max(), bn1_val.min())
            print(bn2_val.max(), bn2_val.min())
            print(bn3_val.max(), bn3_val.min())
            print(bn4_val.max(), bn4_val.min())
            print(bn5_val.max(), bn5_val.min())
            print(bn6_val.max(), bn6_val.min())
            print(debn4_val.max(), debn4_val.min())
            print(debn3_val.max(), debn3_val.min())
            print(debn2_val.max(), debn2_val.min())
            print(debn1_val.max(), debn1_val.min())
            print(recon_ori_vals.max(), recon_ori_vals.min())
            print(reconstruction_vals.max(), reconstruction_vals.min())
            print(loss_G_val, loss_D_val)
            print("=========================================================================")

            if np.isnan(reconstruction_vals.min() ) or np.isnan(reconstruction_vals.max()):
                print("NaN detected!!")
                ipdb.set_trace()

        # Generative Part is updated every iteration
        _, loss_G_val, adv_pos_val, adv_neg_val, loss_recon_val, loss_adv_G_val, reconstruction_vals, recon_ori_vals,bn1_val,bn2_val,bn3_val,bn4_val,bn5_val,bn6_val,debn4_val, debn3_val, debn2_val, debn1_val = sess.run(
                [train_op_G, loss_G, adversarial_pos, adversarial_neg, loss_recon, loss_adv_G, reconstruction, reconstruction_ori, bn1,bn2,bn3,bn4,bn5,bn6,debn4, debn3, debn2, debn1],
                feed_dict={
                    images_tf: images,
                    mask_recon:crops,
                    images_hiding: images_in,
                   # images_global:images_in,
                   # images_hiding: crops,
                    ll:x_t,
 #                   net_size:net_size_tf,		
                    learning_rate: learning_rate_val,
                    is_train: True
                    })

        _, loss_D_val, adv_pos_val, adv_neg_val = sess.run(
                [train_op_D, loss_D, adversarial_pos, adversarial_neg],
                feed_dict={
                    images_tf: images,
                    mask_recon:crops,
                    images_hiding: images_in,
#                    images_global:images_in,
 #                   images_hiding: crops,
                    labels_D:labels_D_hh,
                    ll:x_t,
  #                  net_size:net_size_tf,
                    learning_rate: learning_rate_val/10.,
                    is_train: True
                    })

        print("Iter:", iters, "Gen Loss:", loss_G_val, "Recon Loss:", loss_recon_val, "Gen ADV Loss:", loss_adv_G_val,  "Dis Loss:", loss_D_val, "||||", adv_pos_val.mean(), adv_neg_val.min(), adv_neg_val.max())

        iters += 1

        callbacks = [tensorboard_callback]
        
        

    saver.save(sess, model_path + 'model', global_step=epoch)
#    num=1/(1+0.01*epoch)
 #   learning_rate_val *= num
    learning_rate_val *= 0.99



