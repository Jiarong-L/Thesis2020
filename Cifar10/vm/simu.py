
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
    index0 = np.where(y_train == i)
    index = index0[0]
    # np.random.shuffle(index)
    category_index.append(index)


##############################################################################
'''
Scenario iid.10w.4000
Here get 10 worker nodes.
worker: 10 class * 40 class_data  =400
'''
index_list=[]
worker_nb=10
class_nb=10
class_data=40
for j in range(worker_nb):
    index=np.concatenate([category_index[i][j*class_data:j*class_data+class_data] for i in range(class_nb)])
    index_list.append(index)
len(index_list)
##############################################################################
print(len(index_list))
for i in range(len(index_list)):
    index1=index_list[i]
    print(len(y_train[index1]))
    print(np.unique(y_train[index1],return_counts=True))

##******************************************************************************

base_path='worker_nodes'
workers=['model'+str(int(i+1)) for i in range(worker_nb)] #save like 'worker_nodes/model1/index.npy'
shutil.rmtree(base_path, ignore_errors=True)
for i in range(worker_nb):
    worker=workers[i]
    worker_dir=os.path.join(base_path,worker)
    try:
        os.makedirs(worker_dir)
    except:
        pass
    file_path = os.path.join(worker_dir,'index.npy') ## TODO: copy the training code of the worker to this dir
    np.save(file_path, index_list[i])

index_collection=glob.glob('worker_nodes/*/index.npy')
central_weight_path=os.path.join('central_node','ANN_model.h5')

from models import ANN_model

model=ANN_model()
shutil.rmtree('central_node', ignore_errors=True)
os.makedirs('central_node')
model.save_weights(central_weight_path)