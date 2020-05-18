import pandas as pd
import tensorflow as tf
import keras
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
from keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)
print(tf.test.is_gpu_available())

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_virtual_device_configuration( gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=31000)])



batch_SIZE=300
img_SIZE = 224

val_set = pd.read_pickle('assign/val_set.pkl')
train_set = pd.read_pickle('assign/train_set.pkl')


train_datagen = ImageDataGenerator(rotation_range=360,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   rescale=1 / 255.)

val_datagen = ImageDataGenerator(rescale=1 / 255.)


# Use the dataframe to define train and validation generators
train_generator = train_datagen.flow_from_dataframe(train_set, 
                                                    x_col='dir', 
                                                    y_col='new_label',
                                                    directory = '.',
                                                    target_size=(img_SIZE, img_SIZE),
                                                    batch_size=batch_SIZE,
                                                    class_mode='binary')

val_generator = val_datagen.flow_from_dataframe(val_set, 
                                                  x_col='dir', 
                                                  y_col='new_label',
                                                  directory = '.',
                                                  target_size=(img_SIZE, img_SIZE),
                                                  batch_size=batch_SIZE,
                                                  class_mode='binary')



from models224 import myVGG
model=myVGG()


epo=100
model.summary()

def step_decay(epoch):
    mylr=1e-4
    mylr = mylr/(1+0.05*epoch)
    return mylr
mylr = tf.keras.callbacks.LearningRateScheduler(step_decay)

history= model.fit_generator(train_generator,
                    steps_per_epoch=train_generator.samples // batch_SIZE,
                    epochs=epo,
                    validation_data=val_generator,
                    validation_steps = val_generator.samples // batch_SIZE,
                    callbacks=[mylr])

model.save_weights('myVGG.h5')
   
