import ipdb
import os
from glob import glob
import pandas as pd
import numpy as np
import cv2
from model import *
from util import *
import sys
import argparse

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

#dataset_path=os.path.join(input_name,dataset_type)
dataset_path=input_name
dataset_real_name=dataset_type+"_image"
result_path= os.path.join(output_name,'test_'+dataset_source+'_'+dataset_type+'_'+str(net_size)+'/')
result_path_output= os.path.join(result_path,'images_with_input'+'/')
result_path_hid= os.path.join(result_path,'images_only_hid'+'/')
model_path = os.path.join(model_name,dataset_source+'_'+dataset_type+'_'+str(net_size)+'/')
trainset_path = os.path.join(data_pickle,'protein_trainset_'+dataset_source+'_'+dataset_type+'_size'+str(net_size)+'.pickle')
testset_path  = os.path.join(data_pickle,'protein_testset_'+dataset_source+'_'+dataset_type+'_size'+str(net_size)+'.pickle')
#if pretrained_name is not None:
 #   pretrained_model_path =None
#pretrained_model_path =None
pretrained_model_path =pretrained_name
if pretrained_name is None:
    pretrained_model_path =model_path


learning_rate_val = 0.005
weight_decay_rate = 0.00001
momentum = 0.9
lambda_recon = 0.999
lambda_adv = 0.001
overlap_size = 0
hiding_size = net_size/2

#date=sys.argv[1]
#testset_path  = '../data/protein_testset_2020_12_14_contact.pickle'
#result_path= '../results/contact_test_2020_12_14/'
#dataset_path='../Study/data/protein_contact_masif_filter'
#pretrained_model_path = '../models/protein_contact_2020_12_14'
#trainset_path = os.path.join('../data/','protein_trainset_'+date+'_contact.pickle')
#testset_path  = os.path.join('../data/','protein_testset_'+date+'_contact.pickle')
#dataset_path = '../Study/data/protein_contact_3dcomplex_new/'
#pretrained_model_path = os.path.join('../models/','protein_contact_'+date+'/')
#result_path= os.path.join('../results/','contact_3dcomplex_test_'+date+'/')
if not os.path.exists(result_path):
    os.makedirs( result_path )

if not os.path.exists(result_path_output):
    os.makedirs( result_path_output )

if not os.path.exists(result_path_hid):
    os.makedirs( result_path_hid )

if not os.path.exists( trainset_path ):

    testset_dir = os.path.join( dataset_path, dataset_real_name )
    testset_chain_dir = os.path.join( dataset_path, "length" )
    testset_dir_temp=os.listdir(testset_dir)
    testset_dir_temp.sort()
    testset_chain_dir_temp=os.listdir(testset_chain_dir)
    testset_chain_dir_temp.sort()
    testset = pd.DataFrame({'image_path': map(lambda x: os.path.join( testset_dir, x ), testset_dir_temp),'chain_path':map(lambda x: os.path.join( testset_chain_dir, x ), testset_chain_dir_temp)})

    testset.to_pickle( testset_path )
else:
    testset = pd.read_pickle( trainset_path )

is_train = tf.placeholder( tf.bool )
images_tf = tf.placeholder( tf.float32, [batch_size, net_size, net_size, 3], name="images")
#images_global = tf.placeholder( tf.float32, [batch_size, net_size, net_size, 3], name="images_global")
ll=tf.placeholder( tf.int32, [batch_size])
model = Model()

#reconstruction = model.build_reconstruction(images_tf, is_train)
bn1, bn2, bn3, bn4, bn5, bn6, debn4, debn3, debn2, debn1, reconstruction_ori, reconstruction=model.build_reconstruction(images_tf,ll,net_size, is_train)

# Applying bigger loss for overlapping region
#sess = tf.InteractiveSession()
#
#tf.initialize_all_variables().run()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
model_file=tf.train.latest_checkpoint(pretrained_model_path)
restorer = tf.train.Saver()
restorer.restore( sess, model_file )

#ii = 0
#for start,end in zip(
 #       range(0, len(testset), batch_size),
  #      range(batch_size, len(testset), batch_size)):
#print "%d" %ii
test_image_paths = testset[:batch_size]['image_path'].values
test_chain_paths = testset[:batch_size]['chain_path'].values
test_chain_ori=map(lambda x: load_chain( x,net_size ), test_chain_paths)
test_images_ori = map(lambda x: load_image(x,net_size), test_image_paths)

#    test_images_crop = map(lambda x: crop_random(x, x=32, y=32), test_images_ori)
test_images_crop = map(lambda x,y: crop_interaction(x, y,net_size), test_images_ori,test_chain_ori)
test_images,test_global,test_crops, xs,ys = zip(*test_images_crop)
bn1_val,bn2_val,bn3_val,bn4_val,bn5_val,bn6_val,debn4_val, debn3_val, debn2_val, debn1_val, reconstruction_vals, recon_ori_vals = sess.run(
	[bn1,bn2,bn3,bn4,bn5,bn6,debn4, debn3, debn2, debn1, reconstruction_ori,reconstruction],
        feed_dict={
        	images_tf: test_images,
#                images_hiding: test_crops,
                is_train: False
                })
#    reconstruction_vals = sess.run(
#            reconstruction,
#            feed_dict={
#                images_tf: test_images,
#                images_hiding: test_crops,
#                is_train: False
#                })
iii=0
for rec_val,img,x,y in zip(reconstruction_vals,test_images, xs, ys):
    rec_hid = (255. * rec_val).astype(int)
    rec_con = (255. * img).astype(int)
    rec_1=rec_con.copy()
    xx=net_size-x
#    rec_2=misc.imresize(rec_hid[:,:,0],[y,xx],interp='nearest')
 #   rec_3=misc.imresize(rec_hid[:,:,1],[y,xx],interp='nearest')
  #  rec_4=misc.imresize(rec_hid[:,:,2],[y,xx],interp='nearest')
    rec_hid_temp=rec_1[0:y,x:net_size]
    rec_hid_temp[:,:,0]=rec_hid[0:y,x:net_size,0]
    rec_hid_temp[:,:,1]=rec_hid[0:y,x:net_size,1]
    rec_hid_temp[:,:,2]=rec_hid[0:y,x:net_size,2]
    rec_con[0:y, x:net_size] = rec_hid_temp
#        rec_con[y:y+64, x:x+64] = rec_hid
        #img_rgb = (255. * (img + 1)/2.).astype(int)
#        cv2.imwrite( os.path.join(result_path, 'img_'+str(ii)+'.test.jpg'), rec_con)
    name=test_image_paths[iii]
    name_temp=name.split('/')[-1]
    pdb_name=name_temp.split('.')[0]
    cv2.imwrite( os.path.join(result_path_output, pdb_name+'.jpg'), rec_con)
#    cv2.imwrite( os.path.join(result_path_hid, 'hid_img_'+pdb_name+'_'+str(y)+'_'+str(xx)+'.test.jpg'), rec_hid_temp)
#    cv2.imwrite( os.path.join(result_path, 'img_'+str(ii)+'_'+str(y)+'_'+str(xx)+'.test.jpg'), rec_con)
#    cv2.imwrite( os.path.join(result_path, 'hid_img_'+str(ii)+'_'+str(y)+'_'+str(xx)+'.test.jpg'), rec_hid_temp)
        #cv2.imwrite( os.path.join(result_path, 'img_ori'+str(ii)+'.'+str(int(iters/1000))+'.jpg'), rec_con)
    iii += 1
#        if ii > 30: break

