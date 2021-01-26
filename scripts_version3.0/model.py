#import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np

#import cPickle
import pickle as cPickle
import ipdb
config=tf.ConfigProto()
#config=tf.compat.v1.ConfigProto() # tensorflow1 -> tensorflow2
session=tf.Session(config=config)
#session=tf.compat.v1.Session(config=config) # tensorflow1 -> tensorflow2
class Model():
    def __init__(self):
        pass

    def new_conv_layer( self, bottom, filter_shape, activation=tf.identity, padding='SAME', stride=1, name=None ):
        with tf.variable_scope( name ):
            w = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=filter_shape[-1],
                    initializer=tf.constant_initializer(0.))

            conv = tf.nn.conv2d( bottom, w, [1,stride,stride,1], padding=padding)
            bias = activation(tf.nn.bias_add(conv, b))

        return bias #relu

    def new_deconv_layer(self, bottom, filter_shape, output_shape, activation=tf.identity, padding='SAME', stride=1, name=None):
        with tf.variable_scope(name):
            W = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=filter_shape[-2],
                    initializer=tf.constant_initializer(0.))
            deconv = tf.nn.conv2d_transpose( bottom, W, output_shape, [1,stride,stride,1], padding=padding)
            bias = activation(tf.nn.bias_add(deconv, b))

        return bias

    def new_fc_layer( self, bottom, output_size, name ):
        shape = bottom.get_shape().as_list()
        dim = np.prod( shape[1:] )
        x = tf.reshape( bottom, [-1, dim])
        input_size = dim

        with tf.variable_scope(name):
            w = tf.get_variable(
                    "W",
                    shape=[input_size, output_size],
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=[output_size],
                    initializer=tf.constant_initializer(0.))
            fc = tf.nn.bias_add( tf.matmul(x, w), b)

        return fc

    def channel_wise_fc_layer(self, input, name): # bottom: (7x7x512)
        _, width, height, n_feat_map = input.get_shape().as_list()
        input_reshape = tf.reshape( input, [-1, width*height, n_feat_map] )
        input_transpose = tf.transpose( input_reshape, [2,0,1] )

        with tf.variable_scope(name):
            W = tf.get_variable(
                    "W",
                    shape=[n_feat_map,width*height, width*height], # (512,49,49)
                    initializer=tf.random_normal_initializer(0., 0.005))
            output = tf.batch_matmul(input_transpose, W)

        output_transpose = tf.transpose(output, [1,2,0])
        output_reshape = tf.reshape( output_transpose, [-1, height, width, n_feat_map] )

        return output_reshape

    def leaky_relu(self, bottom, leak=0.1):
        return tf.maximum(leak*bottom, bottom)

    def batchnorm(self, bottom, is_train, epsilon=1e-8, name=None):
        bottom = tf.clip_by_value( bottom, -100., 100.)
        depth = bottom.get_shape().as_list()[-1]

        with tf.variable_scope(name):

            gamma = tf.get_variable("gamma", [depth], initializer=tf.constant_initializer(1.))
            beta  = tf.get_variable("beta" , [depth], initializer=tf.constant_initializer(0.))

            batch_mean, batch_var = tf.nn.moments(bottom, [0,1,2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)


            def update():
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
	    
            with tf.variable_scope(name_or_scope=tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
                    ema_apply_op = ema.apply([batch_mean, batch_var])
                    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
            mean, var = tf.cond(
                    is_train,
                    update,
                    lambda: (ema_mean, ema_var) )

            normed = tf.nn.batch_norm_with_global_normalization(bottom, mean, var, beta, gamma, epsilon, False)
        return normed

#    def build_reconstruction( self, images,images_global,ll,net_size,is_train ):
#        if net_size==128:
#            t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12=build_reconstruction_128( self, images,images_global,is_train )
#            return t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12
#        elif net_size==256:
#            t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12=build_reconstruction_256( self, images,images_global,is_train )
#            return t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12

#    def build_adversarial(self, images,net_size, is_train, reuse=None):
#        if net_size==128:
#            s=build_adversarial_128(self, images,is_train, reuse=None)
#            return s
#        elif net_size==256:
#            s=build_adversarial_256(self, images,is_train, reuse=None)
#            return s

#    def build_adversarial_temp(self, images,net_size, is_train, reuse=None):
#        if net_size==128:
#            s=build_adversarial_temp_128(self, images,is_train, reuse=None)
#            return s
#        elif net_size==256:
#            s=build_adversarial_temp_256(self, images,is_train, reuse=None)
#            return s

    def build_reconstruction_128(self, images,is_train):
        batch_size = images.get_shape().as_list()[0]

        with tf.variable_scope('GEN'):
            conv1 = self.new_conv_layer(images, [4,4,3,64], stride=2, name="conv1" )
            bn1 = self.leaky_relu(self.batchnorm(conv1, is_train, name='bn1'))
            conv2 = self.new_conv_layer(bn1, [4,4,64,64], stride=2, name="conv2" )
            bn2 = self.leaky_relu(self.batchnorm(conv2, is_train, name='bn2'))
            conv3 = self.new_conv_layer(bn2, [4,4,64,128], stride=2, name="conv3")
            bn3 = self.leaky_relu(self.batchnorm(conv3, is_train, name='bn3'))
            conv4 = self.new_conv_layer(bn3, [4,4,128,256], stride=2, name="conv4")
            bn4 = self.leaky_relu(self.batchnorm(conv4, is_train, name='bn4'))
            conv5 = self.new_conv_layer(bn4, [4,4,256,512], stride=2, name="conv5")
            bn5 = self.leaky_relu(self.batchnorm(conv5, is_train, name='bn5'))
            conv6 = self.new_conv_layer(bn5, [4,4,512,4000], stride=2, padding='VALID', name='conv6')
            bn6 = self.leaky_relu(self.batchnorm(conv6, is_train, name='bn6'))

            deconv4 = self.new_deconv_layer( bn6, [4,4,512,4000], conv5.get_shape().as_list(), padding='VALID', stride=2, name="deconv4")
            debn4 = tf.nn.relu(self.batchnorm(deconv4, is_train, name='debn4'))
            deconv3 = self.new_deconv_layer( debn4, [4,4,256,512], conv4.get_shape().as_list(), stride=2, name="deconv3")
            debn3 = tf.nn.relu(self.batchnorm(deconv3, is_train, name='debn3'))
            deconv2 = self.new_deconv_layer( debn3, [4,4,128,256], conv3.get_shape().as_list(), stride=2, name="deconv2")
            debn2 = tf.nn.relu(self.batchnorm(deconv2, is_train, name='debn2'))
            deconv1 = self.new_deconv_layer( debn2, [4,4,64,128], conv2.get_shape().as_list(), stride=2, name="deconv1")
            debn1 = tf.nn.relu(self.batchnorm(deconv1, is_train, name='debn1'))
            deconv1_0 = self.new_deconv_layer( debn1, [4,4,64,64], conv1.get_shape().as_list(), stride=2, name="deconv1_0")
            debn1_0 = tf.nn.relu(self.batchnorm(deconv1_0, is_train, name='debn1_0'))
            recon = self.new_deconv_layer( debn1_0, [4,4,3,64], [batch_size,128,128,3], stride=2, name="recon")
           # reconstruction_ori=tf.nn.tanh(recon)
           # for ii in range(batch_size):
           #     rec_val1=reconstruction_ori[ii,:,:,:]
           #     img1=images_global[ii,:,:,:]
           #     x=ll[ii]
           #     y=x
           #     sess=tf.Session()
           #     rec_val=rec_val1.eval(session=session)
           #     img=img1.eval(session=sess)
           #     rec_hid = rec_val.reshape((64,64,3))
           #     rec_con = img.reshape((128,128,3))
           #     rec_1=rec_con.copy()
            #    xx=128-x
             #   rec_2=misc.imresize(rec_hid[:,:,0],[y,xx],interp='nearest')
             #   rec_3=misc.imresize(rec_hid[:,:,1],[y,xx],interp='nearest')
             #   rec_4=misc.imresize(rec_hid[:,:,2],[y,xx],interp='nearest')
             #   rec_hid_temp=rec_1[0:y,x:128]
             #   rec_hid_temp[:,:,0]=rec_2
             #   rec_hid_temp[:,:,1]=rec_3
             #   rec_hid_temp[:,:,2]=rec_4
             #   rec_con[0:y, x:128] = rec_hid_temp
             #   bb=re_con.reshape((1,128,128,3))
             #   reconstruction_global[ii,:,:,:]=bb
#        return bn1, bn2, bn3, bn4, bn5, bn6, debn4, debn3, debn2, debn1, recon, tf.nn.tanh(recon)
        return bn1, bn2, bn3, bn4, bn5, bn6, debn4, debn3, debn2, debn1, recon, tf.nn.sigmoid(recon)

    def build_adversarial_128(self, images, is_train, reuse=None):
        with tf.variable_scope('DIS', reuse=reuse):
            conv1 = self.new_conv_layer(images, [4,4,3,64], stride=2, name="conv1" )
            bn1 = self.leaky_relu(self.batchnorm(conv1, is_train, name='bn1'))
            conv2 = self.new_conv_layer(bn1, [4,4,64,128], stride=2, name="conv2")
            bn2 = self.leaky_relu(self.batchnorm(conv2, is_train, name='bn2'))
            conv3 = self.new_conv_layer(bn2, [4,4,128,256], stride=2, name="conv3")
            bn3 = self.leaky_relu(self.batchnorm(conv3, is_train, name='bn3'))
            conv4 = self.new_conv_layer(bn3, [4,4,256,512], stride=2, name="conv4")
            bn4 = self.leaky_relu(self.batchnorm(conv4, is_train, name='bn4'))

            output = self.new_fc_layer( bn4, output_size=1, name='output')

        return output[:,0]

    def build_adversarial_temp_128(self, images, is_train, reuse=None):
        with tf.variable_scope('DIS', reuse=reuse):
            conv1 = self.new_conv_layer(images, [4,4,3,64], stride=2, name="conv1_temp" )
            bn1 = self.leaky_relu(self.batchnorm(conv1, is_train, name='bn1_temp'))
            conv2 = self.new_conv_layer(bn1, [4,4,64,128], stride=2, name="conv2_temp")
            bn2 = self.leaky_relu(self.batchnorm(conv2, is_train, name='bn2_temp'))
            conv3 = self.new_conv_layer(bn2, [4,4,128,256], stride=2, name="conv3_temp")
            bn3 = self.leaky_relu(self.batchnorm(conv3, is_train, name='bn3_temp'))
            conv4 = self.new_conv_layer(bn3, [4,4,256,512], stride=2, name="conv4_temp")
            bn4 = self.leaky_relu(self.batchnorm(conv4, is_train, name='bn4_temp'))
            conv5 = self.new_conv_layer(bn4, [4,4,512,512], stride=2, name="conv5_temp")
            bn5 = self.leaky_relu(self.batchnorm(conv5, is_train, name='bn5_temp'))
            output = self.new_fc_layer( bn5, output_size=1, name='output_temp')

        return output[:,0]

    def build_reconstruction_256(self, images,is_train):
        batch_size = images.get_shape().as_list()[0]

        with tf.variable_scope('GEN'):
            conv0 = self.new_conv_layer(images, [4,4,3,32], stride=2, name="conv0-256" )
            bn0 = self.leaky_relu(self.batchnorm(conv0, is_train, name='bn0-256'))
            conv1 = self.new_conv_layer(bn0, [4,4,32,64], stride=2, name="conv1-256" )
            bn1 = self.leaky_relu(self.batchnorm(conv1, is_train, name='bn1-256'))
            conv2 = self.new_conv_layer(bn1, [4,4,64,64], stride=2, name="conv2-256" )
            bn2 = self.leaky_relu(self.batchnorm(conv2, is_train, name='bn2-256'))
            conv3 = self.new_conv_layer(bn2, [4,4,64,128], stride=2, name="conv3-256")
            bn3 = self.leaky_relu(self.batchnorm(conv3, is_train, name='bn3-256'))
            conv4 = self.new_conv_layer(bn3, [4,4,128,256], stride=2, name="conv4-256")
            bn4 = self.leaky_relu(self.batchnorm(conv4, is_train, name='bn4-256'))
            conv5 = self.new_conv_layer(bn4, [4,4,256,512], stride=2, name="conv5-256")
            bn5 = self.leaky_relu(self.batchnorm(conv5, is_train, name='bn5-256'))
            conv6 = self.new_conv_layer(bn5, [4,4,512,4000], stride=2, padding='VALID', name='conv6-256')
            bn6 = self.leaky_relu(self.batchnorm(conv6, is_train, name='bn6-256'))

            deconv4 = self.new_deconv_layer( bn6, [4,4,512,4000], conv5.get_shape().as_list(), padding='VALID', stride=2, name="deconv4-256")
            debn4 = tf.nn.relu(self.batchnorm(deconv4, is_train, name='debn4-256'))
            deconv3 = self.new_deconv_layer( debn4, [4,4,256,512], conv4.get_shape().as_list(), stride=2, name="deconv3")
            debn3 = tf.nn.relu(self.batchnorm(deconv3, is_train, name='debn3-256'))
            deconv2 = self.new_deconv_layer( debn3, [4,4,128,256], conv3.get_shape().as_list(), stride=2, name="deconv2-256")
            debn2 = tf.nn.relu(self.batchnorm(deconv2, is_train, name='debn2-256'))
            deconv1 = self.new_deconv_layer( debn2, [4,4,64,128], conv2.get_shape().as_list(), stride=2, name="deconv1-256")
            debn1 = tf.nn.relu(self.batchnorm(deconv1, is_train, name='debn1-256'))
            deconv0 = self.new_deconv_layer( debn1, [4,4,64,64], conv1.get_shape().as_list(), stride=2, name="deconv0-256")
            debn0 = tf.nn.relu(self.batchnorm(deconv0, is_train, name='debn0-256'))
            deconv0_0 = self.new_deconv_layer( debn0, [4,4,32,64], conv0.get_shape().as_list(), stride=2, name="deconv0_0-256")
            debn0_0 = tf.nn.relu(self.batchnorm(deconv0_0, is_train, name='debn0_0-256'))
            recon = self.new_deconv_layer( debn0_0, [4,4,3,32], [batch_size,256,256,3], stride=2, name="recon-256")
        return bn1, bn2, bn3, bn4, bn5, bn6, debn4, debn3, debn2, debn1, recon, tf.nn.sigmoid(recon)

    def build_adversarial_256(self, images, is_train, reuse=None):
        with tf.variable_scope('DIS', reuse=reuse):
            conv0 = self.new_conv_layer(images, [4,4,3,32], stride=2, name="conv0-256" )
            bn0 = self.leaky_relu(self.batchnorm(conv0, is_train, name='bn0-256'))
            conv1 = self.new_conv_layer(bn0, [4,4,32,64], stride=2, name="conv1-256" )
            bn1 = self.leaky_relu(self.batchnorm(conv1, is_train, name='bn1-256'))
            conv2 = self.new_conv_layer(bn1, [4,4,64,128], stride=2, name="conv2-256")
            bn2 = self.leaky_relu(self.batchnorm(conv2, is_train, name='bn2-256'))
            conv3 = self.new_conv_layer(bn2, [4,4,128,256], stride=2, name="conv3-256")
            bn3 = self.leaky_relu(self.batchnorm(conv3, is_train, name='bn3-256'))
            conv4 = self.new_conv_layer(bn3, [4,4,256,512], stride=2, name="conv4-256")
            bn4 = self.leaky_relu(self.batchnorm(conv4, is_train, name='bn4-256'))

            output = self.new_fc_layer( bn4, output_size=1, name='output-256')

        return output[:,0]

    def build_adversarial_temp_256(self, images, is_train,reuse=None):
        with tf.variable_scope('DIS', reuse=reuse):
            conv0 = self.new_conv_layer(images, [4,4,3,32], stride=2, name="conv0_temp-256" )
            bn0 = self.leaky_relu(self.batchnorm(conv0, is_train, name='bn0_temp-256'))
            conv1 = self.new_conv_layer(bn0, [4,4,32,64], stride=2, name="conv1_temp-256" )
            bn1 = self.leaky_relu(self.batchnorm(conv1, is_train, name='bn1_temp-256'))
            conv2 = self.new_conv_layer(bn1, [4,4,64,128], stride=2, name="conv2_temp-256")
            bn2 = self.leaky_relu(self.batchnorm(conv2, is_train, name='bn2_temp-256'))
            conv3 = self.new_conv_layer(bn2, [4,4,128,256], stride=2, name="conv3_temp-256")
            bn3 = self.leaky_relu(self.batchnorm(conv3, is_train, name='bn3_temp-256'))
            conv4 = self.new_conv_layer(bn3, [4,4,256,512], stride=2, name="conv4_temp-256")
            bn4 = self.leaky_relu(self.batchnorm(conv4, is_train, name='bn4_temp-256'))
            conv5 = self.new_conv_layer(bn4, [4,4,512,512], stride=2, name="conv5_temp-256")
            bn5 = self.leaky_relu(self.batchnorm(conv5, is_train, name='bn5_temp-256'))
            output = self.new_fc_layer( bn5, output_size=1, name='output_temp-256')

        return output[:,0]

    def build_reconstruction_512(self, images,is_train):
        batch_size = images.get_shape().as_list()[0]

        with tf.variable_scope('GEN'):
            conv00 = self.new_conv_layer(images, [4,4,3,32], stride=2, name="conv00-256-" )
            bn00 = self.leaky_relu(self.batchnorm(conv00, is_train, name='bn00-256-'))
            conv0 = self.new_conv_layer(bn00, [4,4,32,32], stride=2, name="conv0-256" )
            bn0 = self.leaky_relu(self.batchnorm(conv0, is_train, name='bn0-256'))
            conv1 = self.new_conv_layer(bn0, [4,4,32,64], stride=2, name="conv1-256" )
            bn1 = self.leaky_relu(self.batchnorm(conv1, is_train, name='bn1-256'))
            conv2 = self.new_conv_layer(bn1, [4,4,64,64], stride=2, name="conv2-256" )
            bn2 = self.leaky_relu(self.batchnorm(conv2, is_train, name='bn2-256'))
            conv3 = self.new_conv_layer(bn2, [4,4,64,128], stride=2, name="conv3-256")
            bn3 = self.leaky_relu(self.batchnorm(conv3, is_train, name='bn3-256'))
            conv4 = self.new_conv_layer(bn3, [4,4,128,256], stride=2, name="conv4-256")
            bn4 = self.leaky_relu(self.batchnorm(conv4, is_train, name='bn4-256'))
            conv5 = self.new_conv_layer(bn4, [4,4,256,512], stride=2, name="conv5-256")
            bn5 = self.leaky_relu(self.batchnorm(conv5, is_train, name='bn5-256'))
            conv6 = self.new_conv_layer(bn5, [4,4,512,4000], stride=2, padding='VALID', name='conv6-256')
            bn6 = self.leaky_relu(self.batchnorm(conv6, is_train, name='bn6-256'))

            deconv4 = self.new_deconv_layer( bn6, [4,4,512,4000], conv5.get_shape().as_list(), padding='VALID', stride=2, name="deconv4-256")
            debn4 = tf.nn.relu(self.batchnorm(deconv4, is_train, name='debn4-256'))
            deconv3 = self.new_deconv_layer( debn4, [4,4,256,512], conv4.get_shape().as_list(), stride=2, name="deconv3")
            debn3 = tf.nn.relu(self.batchnorm(deconv3, is_train, name='debn3-256'))
            deconv2 = self.new_deconv_layer( debn3, [4,4,128,256], conv3.get_shape().as_list(), stride=2, name="deconv2-256")
            debn2 = tf.nn.relu(self.batchnorm(deconv2, is_train, name='debn2-256'))
            deconv1 = self.new_deconv_layer( debn2, [4,4,64,128], conv2.get_shape().as_list(), stride=2, name="deconv1-256")
            debn1 = tf.nn.relu(self.batchnorm(deconv1, is_train, name='debn1-256'))
            deconv0 = self.new_deconv_layer( debn1, [4,4,64,64], conv1.get_shape().as_list(), stride=2, name="deconv0-256")
            debn0 = tf.nn.relu(self.batchnorm(deconv0, is_train, name='debn0-256'))
            deconv00 = self.new_deconv_layer( debn0, [4,4,32,64], conv0.get_shape().as_list(), stride=2, name="deconv00-256")
            debn00 = tf.nn.relu(self.batchnorm(deconv00, is_train, name='debn00-256'))
            deconv00_0 = self.new_deconv_layer( debn00, [4,4,32,32], conv00.get_shape().as_list(), stride=2, name="deconv00_0-256")
            debn00_0 = tf.nn.relu(self.batchnorm(deconv00_0, is_train, name='debn00_0-256'))
            recon = self.new_deconv_layer( debn00_0, [4,4,3,32], [batch_size,512,512,3], stride=2, name="recon-256")
        return bn1, bn2, bn3, bn4, bn5, bn6, debn4, debn3, debn2, debn1, recon, tf.nn.sigmoid(recon)

    def build_adversarial_512(self, images, is_train, reuse=None):
        with tf.variable_scope('DIS', reuse=reuse):
            conv00 = self.new_conv_layer(images, [4,4,3,32], stride=2, name="conv00-256" )
            bn00 = self.leaky_relu(self.batchnorm(conv00, is_train, name='bn00-256'))
            conv0 = self.new_conv_layer(bn00, [4,4,32,32], stride=2, name="conv0-256" )
            bn0 = self.leaky_relu(self.batchnorm(conv0, is_train, name='bn0-256'))
            conv1 = self.new_conv_layer(bn0, [4,4,32,64], stride=2, name="conv1-256" )
            bn1 = self.leaky_relu(self.batchnorm(conv1, is_train, name='bn1-256'))
            conv2 = self.new_conv_layer(bn1, [4,4,64,128], stride=2, name="conv2-256")
            bn2 = self.leaky_relu(self.batchnorm(conv2, is_train, name='bn2-256'))
            conv3 = self.new_conv_layer(bn2, [4,4,128,256], stride=2, name="conv3-256")
            bn3 = self.leaky_relu(self.batchnorm(conv3, is_train, name='bn3-256'))
            conv4 = self.new_conv_layer(bn3, [4,4,256,512], stride=2, name="conv4-256")
            bn4 = self.leaky_relu(self.batchnorm(conv4, is_train, name='bn4-256'))

            output = self.new_fc_layer( bn4, output_size=1, name='output-256')

        return output[:,0]

    def build_adversarial_temp_512(self, images, is_train,reuse=None):
        with tf.variable_scope('DIS', reuse=reuse):
            conv00 = self.new_conv_layer(images, [4,4,3,32], stride=2, name="conv00_temp-256" )
            bn00 = self.leaky_relu(self.batchnorm(conv00, is_train, name='bn00_temp-256'))
            conv0 = self.new_conv_layer(bn00, [4,4,32,32], stride=2, name="conv0_temp-256" )
            bn0 = self.leaky_relu(self.batchnorm(conv0, is_train, name='bn0_temp-256'))
            conv1 = self.new_conv_layer(bn0, [4,4,32,64], stride=2, name="conv1_temp-256" )
            bn1 = self.leaky_relu(self.batchnorm(conv1, is_train, name='bn1_temp-256'))
            conv2 = self.new_conv_layer(bn1, [4,4,64,128], stride=2, name="conv2_temp-256")
            bn2 = self.leaky_relu(self.batchnorm(conv2, is_train, name='bn2_temp-256'))
            conv3 = self.new_conv_layer(bn2, [4,4,128,256], stride=2, name="conv3_temp-256")
            bn3 = self.leaky_relu(self.batchnorm(conv3, is_train, name='bn3_temp-256'))
            conv4 = self.new_conv_layer(bn3, [4,4,256,512], stride=2, name="conv4_temp-256")
            bn4 = self.leaky_relu(self.batchnorm(conv4, is_train, name='bn4_temp-256'))
            conv5 = self.new_conv_layer(bn4, [4,4,512,512], stride=2, name="conv5_temp-256")
            bn5 = self.leaky_relu(self.batchnorm(conv5, is_train, name='bn5_temp-256'))
            output = self.new_fc_layer( bn5, output_size=1, name='output_temp-256')

        return output[:,0]

    def build_reconstruction(self, images,ll,net_size,is_train):
        if net_size==128:
            t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12=self.build_reconstruction_128(images,is_train )
            return t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12
        elif net_size==256:
            t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12=self.build_reconstruction_256(images,is_train )
            return t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12
        elif net_size==512:
            t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12=self.build_reconstruction_512(images,is_train )
            return t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12

    def build_adversarial(self, images,net_size, is_train,reuse=None):
        if net_size==128:
            s=self.build_adversarial_128(images,is_train, reuse)
            return s
        elif net_size==256:
            s=self.build_adversarial_256(images,is_train, reuse)
            return s
        elif net_size==512:
            s=self.build_adversarial_512(images,is_train, reuse)
            return s

    def build_adversarial_temp(self, images,net_size, is_train,reuse=None):
        if net_size==128:
            s=self.build_adversarial_temp_128(images,is_train, reuse)
            return s
        elif net_size==256:
            s=self.build_adversarial_temp_256(images,is_train, reuse)
            return s
        elif net_size==512:
            s=self.build_adversarial_temp_512(images,is_train, reuse)
            return s
