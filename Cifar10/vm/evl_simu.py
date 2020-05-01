
import tensorflow as tf
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
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
Scenario iid.10w.10000
Here get 10 worker nodes.
worker: 10 class * 100 class_data  =1000

'''
index_list=[]
worker_nb=10
class_nb=10
class_data=100

class_set=[]
for i in range(worker_nb):
    x=[i+j for j in range(class_nb)]
    for c in range(len(x)):
        while True:
            if x[c]<10:
                break
            x[c]=x[c]-10  
    class_set.append(x)
print(class_set)

record = [0 for i in range(10)]

for j in range(worker_nb):
    index=np.concatenate([category_index[i][record[i]*class_data:record[i]*class_data+class_data] for i in class_set[j]])
    for i in class_set[j]:
        record[i] = record[i]+1
    index_list.append(index)
    print(record)
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

