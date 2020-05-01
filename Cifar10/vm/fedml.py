
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import psutil
import math

from worker_node import node_training_process
from models import ANN_model

@tf.function
def get_nb_matrix(model_weights,nb):
    model_w=[]
    for layer in model_weights:
        layer_w=np.empty(layer.shape)
        layer_w.fill(nb)
        model_w.append(layer_w)
    return model_w


def memory(epo):
    pid = os.getpid()
    memoryUse = psutil.Process(pid).memory_info()[0]
    with open('result/memorylog.txt','a') as file_handle:
        file_handle.write(str(epo))
        file_handle.write('\n')
        file_handle.write(str(memoryUse))
        file_handle.write('\n')
    print('memory use:', memoryUse)


class Fed_Training():
    # This is only a simulation
    
    def __init__(self,model,nodes_path,central_path,x_test, y_test,batch_size = 50,augument=False,local_iid=False,node_evl=False):
        self.model=model
        self.epo=0

        self.nodes_p=nodes_path  # treat it as the index of non-delayed nodes
        self.central_p=central_path
    
        self.x_test=x_test    
        self.y_test=y_test

        self.batch_size=batch_size
        self.augument=augument
        self.local_iid=local_iid
        self.node_evl=node_evl
        self.bad_node=False

        self.acc_h=[]
        self.loss_h=[]
        self.epo_h=[]

        self.delayed_index=[]
        self.delayed_speed=1

        self.shared_index=[]

        self.finetune_index=[]

    def load_test_set(self):
        '''
        Load the testing set
        '''
        autotune=tf.data.experimental.AUTOTUNE
        self.total_testing=self.y_test.shape[0]
        x_te=tf.data.Dataset.from_tensor_slices(self.x_test)
        y_te=tf.data.Dataset.from_tensor_slices(self.y_test)
        self.test_set=tf.data.Dataset.zip((x_te, y_te)).repeat().batch(self.batch_size).prefetch(buffer_size=autotune)
        return True

    def set_shared_index(self,shared_index):
        self.shared_index=shared_index
        return True

    def set_delayed_update(self,delayed_index,delayed_speed):
        self.delayed_index=delayed_index
        self.delayed_speed=delayed_speed
        return True

    def set_bad_node(self,bad_node_nb,bad_node_size):
        self.bad_node_nb=bad_node_nb
        self.bad_node_size=bad_node_size
        if bad_node_nb!=0:
            if bad_node_size!=0:
                self.bad_node=True
        return self.bad_node

    def delayed_node_train_step(self):
        '''
        Upload weights for delayed nodes
        '''
        self.delayed_weight_list=[]
        self.delayed_nb_list=[]
        # g = tf.Graph()
        # with g.as_default(): #tf.graph to solve memory leak
        for index_path in self.delayed_index:
            # Run each nodes and collect their weights
            model_weights, nb=node_training_process(index_path,self.shared_index,self.central_p,self.local_epoch,self.batch_size,self.augument,self.local_iid,self.node_evl)
            model_w=get_nb_matrix(model_weights,nb)
            self.delayed_weight_list.append(model_weights)
            self.delayed_nb_list.append(model_w)
            tf.keras.backend.clear_session() ########## to solve memory leak 1/2


    ## TODO: Save each node's acc and loss and plot / compare with the central node
    def fed_train_step(self):
        '''
        A single communication round (Upload weights for non-delayed nodes)
        '''
        self.weight_list=[]
        self.nb_list=[]

        if self.bad_node:
            print('bad node !!!!! number{}, size{}!!!!!!'.format(self.bad_node_nb,self.bad_node_size))
            for i in range(self.bad_node_nb):
                bad_model = ANN_model()
                bad_weight = bad_model.get_weights()
                bad_model_w=get_nb_matrix(bad_weight,self.bad_node_size)
                self.weight_list.append(bad_weight)
                self.nb_list.append(bad_model_w)

        for index_path in self.nodes_p:
            # Run each nodes and collect their weights
            model_weights, nb=node_training_process(index_path,self.shared_index,self.central_p,self.local_epoch,self.batch_size,self.augument,self.local_iid,self.node_evl)
            model_w=get_nb_matrix(model_weights,nb)
            self.weight_list.append(model_weights)
            self.nb_list.append(model_w)
            tf.keras.backend.clear_session() ########## to solve memory leak 2/2

        self.epo=self.epo+1

        # memory(self.epo)  ## Testing memory usage

        return True

    def average_weight(self):
        '''
        Average node weights and save global model's weights   
        '''
        avg_weight=np.average(self.weight_list, axis=0, weights=self.nb_list)
        self.model.set_weights(avg_weight)
        self.model.save_weights(self.central_p)
        return True

    def evaluate(self):
        '''  
        Calculate and save this global round's acc and loss and epoch
        '''
        [loss,acc] =self.model.evaluate(self.test_set,steps=self.total_testing//self.batch_size,verbose=1)
        self.acc_h.append(acc)
        self.loss_h.append(loss)
        self.epo_h.append(self.epo)

        print('epoch:{}'.format(self.epo_h[-1]))
        return True



    def fit(self,epoch,patience=100,local_epoch=1):
        '''
        Federated training process
        '''
        self.local_epoch=local_epoch
        max_acc=0
        count=0

        # # OLD_EVL_NODE: divide the testing set to each worker node for node_evl
        # if self.node_evl:
        #     evl_nb = len(self.nodes_p)
        #     evl_len = int(math.floor(10000/evl_nb)) # since central node use 10000 testing set
        #     evl_total = np.array([i for i in range(10000)])
        #     np.random.shuffle(evl_total)
        #     for i in range(evl_nb):
        #         base_p = self.nodes_p[i][:-9]+'evl_index.npy'
        #         evl_set = evl_total[i*evl_len:i*evl_len+evl_len]
        #         np.save(base_p,evl_set)


        for ee in range(epoch):
            self.fed_train_step()

            # To control delayed updating
            if len(self.delayed_index)!=0:

                if (self.epo % self.delayed_speed) == 1:

                    if self.epo == 1:
                        self.delayed_node_train_step()
                    else:
                        self.weight_list = self.weight_list+self.delayed_weight_list
                        self.nb_list = self.nb_list + self.delayed_nb_list
                        self.delayed_node_train_step()

            self.average_weight()

            self.evaluate()

            if max_acc < self.acc_h[-1]:   # early stopping, if it doesn't increase for 100 epoch, 
                count = 0                   # TODO: add learning rate controller
                max_acc = self.acc_h[-1]
            else:
                count=count+1
            if count>patience:
                print('break')
                break

        return self.epo_h,self.loss_h,self.acc_h


