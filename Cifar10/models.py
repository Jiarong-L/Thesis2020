import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import glob


# def ANN_model():  # from https://keras.io/examples/cifar10_cnn/
#     model = tf.keras.Sequential()
#     model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',input_shape=(32,32,3)))
#     model.add(tf.keras.layers.Activation('relu'))
#     model.add(tf.keras.layers.Conv2D(32, (3, 3)))
#     model.add(tf.keras.layers.Activation('relu'))
#     model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#     model.add(tf.keras.layers.Dropout(0.25))

#     model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
#     model.add(tf.keras.layers.Activation('relu'))
#     model.add(tf.keras.layers.Conv2D(64, (3, 3)))
#     model.add(tf.keras.layers.Activation('relu'))
#     model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#     model.add(tf.keras.layers.Dropout(0.25))

#     model.add(tf.keras.layers.Flatten())
#     model.add(tf.keras.layers.Dense(512))
#     model.add(tf.keras.layers.Activation('relu'))
#     model.add(tf.keras.layers.Dropout(0.5))
#     model.add(tf.keras.layers.Dense(10))
#     model.add(tf.keras.layers.Activation('softmax'))

#     # initiate RMSprop optimizer
#     opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

#     # Let's train the model using RMSprop
#     model.compile(loss='sparse_categorical_crossentropy',
#                 optimizer=opt,
#                 metrics=['acc'])
#     return model





def ANN_model():  # from mattias
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',input_shape=(32,32,3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(32, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(100))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation('softmax'))
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])

    return model