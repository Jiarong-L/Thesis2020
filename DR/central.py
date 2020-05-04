import pandas as pd
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import cv2

tf.__version__

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration( gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15500)])
#Tesla P100-PCIE-16GB

def load_img(img,lab):
    img=tf.io.read_file(img)
    img=tf.image.decode_jpeg(img, channels=1)
    img = tf.cast(img,tf.float32) ##???
    lab = tf.cast(lab,tf.float32)/2 #####???    turn 0,2,4 to 0,1,2 otherwise loss=NaN
    return img,lab

test_data=pd.read_csv('test.txt',header=None,sep=' ',names=['picture','label'])
train_data=pd.read_csv('train.txt',header=None,sep=' ',names=['picture','label'])
valid_data=pd.read_csv('valid.txt',header=None,sep=' ',names=['picture','label'])

test_data['dir']=[os.path.join('preprocessed','test',pic) for pic in test_data['picture']]
train_data['dir']=[os.path.join('preprocessed','train',pic) for pic in train_data['picture']]
valid_data['dir']=[os.path.join('preprocessed','valid',pic) for pic in valid_data['picture']]

total_set = pd.concat([test_data,train_data,valid_data],ignore_index=True)
iid_total_set = pd.concat([total_set[total_set['label']==i*2][:1000] for i in range(3)])

random_set = iid_total_set.sample(frac=1).reset_index(drop=True)

## settings
fraction_val = 0.2 ############################################################################

valid_LEN=int(len(random_set)*fraction_val)
train_LEN=int(len(random_set)-valid_LEN)

batch_SIZE=100 ####################################################################################
train_EPO=int(train_LEN // batch_SIZE)
valid_EPO=int(valid_LEN // batch_SIZE)

iid_valid = random_set[:valid_LEN]
iid_train = random_set[valid_LEN:]

train_SET = tf.data.Dataset.from_tensor_slices((iid_train['dir'],iid_train['label'])).\
                            shuffle(train_LEN).\
                            map(load_img).\
                            batch(batch_SIZE).\
                            repeat().\
                            prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
valid_SET = tf.data.Dataset.from_tensor_slices((iid_valid['dir'],iid_valid['label'])).\
                            shuffle(valid_LEN).\
                            map(load_img).\
                            batch(batch_SIZE).\
                            repeat().\
                            prefetch(buffer_size=tf.data.experimental.AUTOTUNE)



myshape = (299,299,1)
model = tf.keras.applications.xception.Xception(include_top=True, weights=None, input_shape=myshape, classes=3)
# it said include_top should be false if input_shape is not (229,229,3), but the only difference is to add avg_pooling and prediction
model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['acc'])


epo=100 #5000
history= model.fit(train_SET,
                  epochs=epo,
                  steps_per_epoch=train_EPO,
                  validation_data=valid_SET,
                  validation_steps=valid_EPO
                  )

model.save_weights('myXception.h5')

acc = history.history.get('acc')
val_acc = history.history.get('val_acc')
loss = history.history.get('loss')
val_loss = history.history.get('val_loss')


np.save("acc.npy", acc)
np.save("val_acc.npy", val_acc)
np.save("loss.npy", loss)
np.save("val_loss.npy", val_loss)