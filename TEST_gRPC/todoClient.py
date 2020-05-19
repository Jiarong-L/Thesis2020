from concurrent import futures
import logging

import grpc

import todo_pb2_grpc
import todo_pb2



import tensorflow as tf
import pandas as pd
import numpy as np
import os
import shutil
import glob
import base64
import math

from models224 import myVGG
from keras.preprocessing.image import ImageDataGenerator
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_virtual_device_configuration( gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=31000)])


WEIGHT_PATH="client001/client_weights.h5"
TRAIN_DATA = pd.read_pickle('index/worker1.pkl')
CLIENTSIZE=int(len(TRAIN_DATA))



def ClientgetWeight(stub):
    '''get weight from server'''

    myrequest=todo_pb2.void()

    msg=''
    for res in stub.getWeight(myrequest):
        msg=msg+res.model

    with open(WEIGHT_PATH,"wb") as file:
        file.write(base64.b64decode(msg))
    
    return 'client saved weight'



def iter_weight(encoded_string):
    MAX_MESSAGE=100000
    step=len(encoded_string)/MAX_MESSAGE
    step=math.ceil(step) 
    for i in range(step):
        msg=encoded_string[i*MAX_MESSAGE:(i+1)*MAX_MESSAGE]
        yield todo_pb2.modelWeight(model=msg,clientname=WEIGHT_PATH,clientsize=CLIENTSIZE)   



def ClientsendWeight(stub):
    '''send weight to server'''

    with open(WEIGHT_PATH,'rb') as file:
        encoded_string = base64.b64encode(file.read())

    myIter=iter_weight(encoded_string)
    res = stub.sendWeight(myIter)

    return res

def ClientreCheck(stub): # Not USED
    myrequest=todo_pb2.void()
    res = stub.reCheck(myrequest)

    return res



def run(epo_nb):

    MAX_MESSAGE_LENGTH=100000+50 
    SETTINGS=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)]

    with grpc.insecure_channel('39.104.81.105:50051', options=SETTINGS) as channel: #localhost
        stub = todo_pb2_grpc.FedMLStub(channel)
        msg=ClientgetWeight(stub)
        print('ClientgetWeight done')




        #---------------Client do local training here, save weight at WEIGHT_PATH---------------------------
        g = tf.Graph()
        with g.as_default(): #tf.graph to solve memory leak
            
            model=myVGG(epo_nb)
            model.load_weights(WEIGHT_PATH)

            batch_SIZE=300
            train_datagen = ImageDataGenerator(rotation_range=360,horizontal_flip=True,vertical_flip=True,rescale=1 / 255.)
            train_generator = train_datagen.flow_from_dataframe(TRAIN_DATA, 
                                                    x_col='dir', 
                                                    y_col='new_label',
                                                    directory = '.',
                                                    target_size=(224, 224),
                                                    batch_size=batch_SIZE,
                                                    class_mode='binary')
            epo = 1
            history= model.fit_generator(train_generator,
                                        steps_per_epoch=train_generator.samples // batch_SIZE,
                                        epochs=epo)

            model.save_weights(WEIGHT_PATH)      #save model locally
            filename = WEIGHT_PATH[:-17]+'self_EVAL.txt'    #save history locally
            with open(filename,'a') as f:
                f.write(str(history.history.get('loss')))
                f.write(' ')
                f.write(str(history.history.get('acc')))
                f.write('\n')
            del model
        tf.keras.backend.clear_session()

        # -----------------------------------local training ends--------------------------



        msg=ClientsendWeight(stub) 
        print('ClientsendWeight done')

        # msg=ClientreCheck(stub)
        return msg
    

if __name__ == '__main__':
    logging.basicConfig()
    for i in range(100):
        a=run(i)
        print('Round over')