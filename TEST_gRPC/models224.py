import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import os
import matplotlib.pyplot as plt
import glob


def myVGG(epo_nb):

    basemodel = tf.keras.applications.VGG16(input_shape=(224,224,3), include_top=False, weights='imagenet')
        
    # for layer in basemodel.layers[:6]:
    #     layer.trainable = False

    # for layer in basemodel.layers[6:]:
    #     layer.trainable = True
        
    last_layer = basemodel.get_layer('block5_pool')
    last_output = last_layer.output
        
    x = tf.keras.layers.Flatten()(last_output)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(basemodel.input, x)

    mylr=1e-4
    mylr = mylr/(1+0.05*epo_nb)
    print('learning rate:{}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'.format(mylr))

    # opt = tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9)
    opt = tf.keras.optimizers.Adam(lr=mylr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.05)

    model.compile(loss='binary_crossentropy',
                optimizer=opt,
                metrics=['acc'])


    return model