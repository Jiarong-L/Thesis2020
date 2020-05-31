
import tensorflow as tf
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import glob

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()  #50000 train, 10000 test

category_index=[]     # get a list of turples, each turple is the set of a class i.e. total_set[class_nb][img/label]

for i in range(10):
    index0 = np.where(y_test == i)
    index = index0[0]
    # np.random.shuffle(index)
    category_index.append(index)


##############################################################################
'''
unbanlanced testing set

'''
index_list=[]
worker_nb = 10


index = category_index[0][:900]
index_list.append(index)

index = category_index[1][:800]
index_list.append(index)

index = np.concatenate([category_index[2][:500],category_index[3][:300],category_index[4][:200]])
index_list.append(index)

mix = np.concatenate([category_index[i+5] for i in range(5)])
mix = np.concatenate([mix, category_index[0][900:],category_index[1][800:],category_index[2][500:],category_index[3][300:],category_index[4][200:]])
np.random.shuffle(mix)
print(np.unique(y_test[mix],return_counts=True))


record = 0
for i in range(6):
    class_data=random.randint(600,900)
    index = mix[record:record+class_data]
    record = record+class_data
    index_list.append(index)

index = mix[record:]
index_list.append(index)

len(index_list)
##############################################################################
print(len(index_list))
for i in range(len(index_list)):
    index1=index_list[i]
    print(len(y_test[index1]))
    print(np.unique(y_test[index1],return_counts=True))

##******************************************************************************

base_path='worker_nodes'
workers=['model'+str(int(i+1)) for i in range(worker_nb)] #save like 'worker_nodes/model1/index.npy'

for i in range(worker_nb):
    worker=workers[i]
    worker_dir=os.path.join(base_path,worker)

    file_path = os.path.join(worker_dir,'evl_index.npy') ## TODO: copy the training code of the worker to this dir
    np.save(file_path, index_list[i])

