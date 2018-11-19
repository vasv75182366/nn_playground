#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import keras.backend as K
from keras.datasets import cifar10
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, BatchNormalization, MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils


# In[2]:


from binary_ops import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense, BinaryConv2D


def binary_tanh(x):
    return binary_tanh_op(x)


H = 1.
kernel_lr_multiplier = 'Glorot'

# nn
batch_size = 50
epochs = 20
channels = 3
img_rows = 32 
img_cols = 32 
filters = 32 
kernel_size = (3, 3)
pool_size = (2, 2)
hidden_units = 128
classes = 10
use_bias = False

# learning rate schedule
lr_start = 1e-3
lr_end = 1e-4
lr_decay = (lr_end / lr_start)**(1. / epochs)

# BN
epsilon = 1e-6
momentum = 0.9

# dropout
p1 = 0.25
p2 = 0.5

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape)

X_train = X_train.reshape(50000, 3, 32, 32)
X_test = X_test.reshape(10000, 3, 32, 32)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# In[3]:


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, classes) * 2 - 1 # -1 or 1 for hinge loss
Y_test = np_utils.to_categorical(y_test, classes) * 2 - 1


# In[4]:


model = Sequential()

# conv1
model.add(BinaryConv2D(128, kernel_size=kernel_size, input_shape=(channels, img_rows, img_cols),
                       data_format='channels_first',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       padding='same', use_bias=use_bias, name='conv1'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn1'))
model.add(Activation(binary_tanh, name='act1'))


# In[5]:


# conv2
model.add(BinaryConv2D(128, kernel_size=kernel_size,
                       data_format='channels_first',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       padding='same', use_bias=use_bias, name='conv2'))
model.add(MaxPooling2D(pool_size=pool_size, name='pool2', data_format='channels_first'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn2'))
model.add(Activation(binary_tanh, name='act2'))


# In[6]:


# conv3
model.add(BinaryConv2D(256, kernel_size=kernel_size,
                       data_format='channels_first',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       padding='same', use_bias=use_bias, name='conv3'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn3'))
model.add(Activation(binary_tanh, name='act3'))


# In[7]:


# conv4
model.add(BinaryConv2D(256, kernel_size=kernel_size,
                       data_format='channels_first',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       padding='same', use_bias=use_bias, name='conv4'))
model.add(MaxPooling2D(pool_size=pool_size, name='pool4', data_format='channels_first'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn4'))
model.add(Activation(binary_tanh, name='act4'))


# In[8]:


# conv5
model.add(BinaryConv2D(512, kernel_size=kernel_size,
                       data_format='channels_first',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       padding='same', use_bias=use_bias, name='conv5'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn5'))
model.add(Activation(binary_tanh, name='act5'))


# In[9]:


# conv6
model.add(BinaryConv2D(512, kernel_size=kernel_size,
                       data_format='channels_first',
                       H=H, kernel_lr_multiplier=kernel_lr_multiplier, 
                       padding='same', use_bias=use_bias, name='conv6'))
model.add(MaxPooling2D(pool_size=pool_size, name='pool6', data_format='channels_first'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, axis=1, name='bn6'))
model.add(Activation(binary_tanh, name='act6'))


# In[10]:



model.add(Flatten())
# dense1
model.add(BinaryDense(1024, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense7'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn7'))
model.add(Activation(binary_tanh, name='act7'))
# dense2
model.add(BinaryDense(1024, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense8'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn8'))
model.add(Activation(binary_tanh, name='act8'))
# dense3
model.add(BinaryDense(classes, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias, name='dense9'))
model.add(BatchNormalization(epsilon=epsilon, momentum=momentum, name='bn9'))


# In[11]:


opt = Adam(lr=lr_start) 
model.compile(loss='squared_hinge', optimizer=opt, metrics=['acc'])
model.summary()


# In[14]:


epochs = 50
lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=1, validation_data=(X_test, Y_test),
                    callbacks=[lr_scheduler])
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


# In[24]:


verbose = 0
import json
arch = model.to_json()
arch = json.loads(arch)

with open('./binarynet_CIFAR_dumped.nnet', 'w') as fout:
    fout.write('layers ' + str(len(model.layers)) + '\n')

    layers = []
    for ind, l in enumerate(arch["config"]):
        if verbose:
            print(ind, l)
        fout.write('layer ' + str(ind) + ' ' + l['class_name'] + '\n')

        if verbose:
            print(str(ind), l['class_name'])
        layers += [l['class_name']]
        if l['class_name'] == 'Conv2D':
            #fout.write(str(l['config']['nb_filter']) + ' ' + str(l['config']['nb_col']) + ' ' + str(l['config']['nb_row']) + ' ')

            #if 'batch_input_shape' in l['config']:
            #    fout.write(str(l['config']['batch_input_shape'][1]) + ' ' + str(l['config']['batch_input_shape'][2]) + ' ' + str(l['config']['batch_input_shape'][3]))
            #fout.write('\n')

            W = model.layers[ind].get_weights()[0]
            if verbose:
                print(W.shape)
            fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + ' ' + str(W.shape[2]) + ' ' + str(W.shape[3]) + ' ' + l['config']['padding'] + '\n')

            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    for k in range(W.shape[2]):
                        fout.write(str(W[i,j,k]) + '\n')
            fout.write(str(model.layers[ind].get_weights()[1]) + '\n')
###
        if l['class_name'] == 'BinaryConv2D':
            #fout.write(str(l['config']['nb_filter']) + ' ' + str(l['config']['nb_col']) + ' ' + str(l['config']['nb_row']) + ' ')

            #if 'batch_input_shape' in l['config']:
            #    fout.write(str(l['config']['batch_input_shape'][1]) + ' ' + str(l['config']['batch_input_shape'][2]) + ' ' + str(l['config']['batch_input_shape'][3]))
            #fout.write('\n')

            W = model.layers[ind].get_weights()[0]
            if verbose:
                print(W.shape)
            fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + ' ' + str(W.shape[2]) + ' ' + str(W.shape[3]) + ' ' + l['config']['padding'] + '\n')
            for k in range(W.shape[3]):
                for c in range(W.shape[2]):
                    for j in range(W.shape[1]):
                        for i in range(W.shape[0]):
                            fout.write(str(W[i,j,c,k]) + ' ')
                    fout.write('\n')
            # fout.write(str(model.layers[ind].get_weights()[1]) + '\n') #BinaryConv2D has no bias
###
        if l['class_name'] == 'Activation':
            fout.write(l['config']['activation'] + '\n')
        if l['class_name'] == 'MaxPooling2D':
            fout.write(str(l['config']['pool_size'][0]) + ' ' + str(l['config']['pool_size'][1]) + '\n')
        #if l['class_name'] == 'Flatten':
        #    print(l['config']['name'])
        if l['class_name'] == 'Dense':
            #fout.write(str(l['config']['output_dim']) + '\n')
            W = model.layers[ind].get_weights()[0]
            if verbose:
                print(W.shape)
            fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + '\n')


            for w in W:
                fout.write(str(w) + '\n')
            fout.write(str(model.layers[ind].get_weights()[1]) + '\n')
        if l['class_name'] == 'BinaryDense':
            #fout.write(str(l['config']['output_dim']) + '\n')
            W = model.layers[ind].get_weights()[0]
            if verbose:
                print(W.shape)
            fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + '\n')


            for w in W:
                fout.write(str(w) + '\n')
            #fout.write(str(model.layers[ind].get_weights()[1]) + '\n') #BinaryDense has no bias
        if l['class_name'] == 'BatchNormalization':
            #fout.write(str(l['config']['output_dim']) + '\n')
            
            if verbose:
                print(W.shape)
            fout.write(str(model.layers[ind].get_weights()[0].shape) + ' ' +
                       str(model.layers[ind].get_weights()[1].shape) + ' ' +
                       str(model.layers[ind].get_weights()[2].shape) + ' ' +
                       str(model.layers[ind].get_weights()[3].shape) + ' ' + '\n')

            fout.write(str(model.layers[ind].get_weights()[0]) + '\n')
            fout.write(str(model.layers[ind].get_weights()[1]) + '\n')
            fout.write(str(model.layers[ind].get_weights()[2]) + '\n')
            fout.write(str(model.layers[ind].get_weights()[3]) + '\n')


# In[42]:


verbose = 0
import json
import struct

arch = model.to_json()
arch = json.loads(arch)

file_count = -1

    
for ind, l in enumerate(arch["config"]):
    
    if l['class_name'] == 'BinaryConv2D':
        file_count += 1
        W = model.layers[ind].get_weights()[0]

        with open('./cifar_param/array' + str(file_count), 'w') as fout:
            for k in range(W.shape[3]):
                for c in range(W.shape[2]):
                    for j in range(W.shape[1]):
                        for i in range(W.shape[0]):
                            fout.write(struct.pack('<%df' % 1, W[i,j,c,k]))


    if l['class_name'] == 'BinaryDense':
        file_count += 1
        W = model.layers[ind].get_weights()[0]
        
        with open('./cifar_param/array' + str(file_count), 'w') as fout:
            for k in range(W.shape[1]):
                for i in range(W.shape[0]):
                    fout.write(struct.pack('<%df' % 1, W[i,k]))

    if l['class_name'] == 'BatchNormalization':
        file_count += 1
        with open('./cifar_param/array' + str(file_count), 'w') as fout_k:
            file_count += 1
            with open('./cifar_param/array' + str(file_count), 'w') as fout_h:
                for i in range(model.layers[ind].get_weights()[0].shape[0]):
                    beta = model.layers[ind].get_weights()[0][i]
                    gamma = model.layers[ind].get_weights()[1][i]
                    mean = model.layers[ind].get_weights()[2][i]
                    variance = model.layers[ind].get_weights()[3][i]

                    k = gamma / np.sqrt(np.square(variance) + epsilon)
                    h = beta - mean * k
                    fout_k.write(struct.pack('<%df' % 1, k))
                    fout_h.write(struct.pack('<%df' % 1, h))

