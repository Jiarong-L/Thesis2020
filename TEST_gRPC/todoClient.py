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

from models import ANN_model,Xception_model


WEIGHT_PATH="client001/client_weights.h5"
TRAIN_DATA = pd.read_pickle('index/worker1.pkl')
CLIENTSIZE=int(len(TRAIN_DATA))


def load_img(img,lab):
    img=tf.io.read_file(img)
    img=tf.image.decode_jpeg(img, channels=1)
    img = tf.cast(img,tf.float32) 
    lab = tf.cast(lab,tf.float32)/2 
    return img,lab


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

def ClientreCheck(stub):
    myrequest=todo_pb2.void()
    res = stub.reCheck(myrequest)

    return res



def run():

    MAX_MESSAGE_LENGTH=100000+50  #TODO: still not enough to shift my model...should I shift model by layer??
    SETTINGS=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)]

    with grpc.insecure_channel('39.104.16.170:50051', options=SETTINGS) as channel: #localhost
        stub = todo_pb2_grpc.FedMLStub(channel)
        msg=ClientgetWeight(stub)
        print('ClientgetWeight done')




        #TODO: Client do local training here, save weight at WEIGHT_PATH
        g = tf.Graph()
        with g.as_default(): #tf.graph to solve memory leak
            myshape = (229,229,1)
            model=Xception_model(myshape)
            model.load_weights(WEIGHT_PATH)

            batch_SIZE=100
            train_LEN = int(len(TRAIN_DATA))
            train_EPO=int(train_LEN // batch_SIZE)
            train_SET = tf.data.Dataset.from_tensor_slices((TRAIN_DATA['dir'],TRAIN_DATA['label'])).\
                                            shuffle(train_LEN).\
                                            map(load_img).\
                                            batch(batch_SIZE).\
                                            repeat().\
                                            prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            epo=1
            history = model.fit(train_SET,epochs=epo,steps_per_epoch=train_EPO)
            model.save_weights(WEIGHT_PATH)
            filename = WEIGHT_PATH[:-17]+'self_EVAL.txt'
            with open(filename,'a') as f:
                f.write(str(history.history.get('loss')))
                f.write(' ')
                f.write(str(history.history.get('acc')))
                f.write('\n')

            del model
        tf.keras.backend.clear_session()





        msg=ClientsendWeight(stub) 
        print('ClientsendWeight done')

        # msg=ClientreCheck(stub)
        return msg
    

if __name__ == '__main__':
    logging.basicConfig()
    for i in range(500):
        a=run()
        print('Round over')