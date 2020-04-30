from concurrent import futures
import logging

import grpc

import todo_pb2_grpc
import todo_pb2



import tensorflow as tf
import numpy as np
import os
import shutil
import glob
import base64
import math

from models import ANN_model
model=ANN_model()

WEIGHT_PATH="client002/test_weights.h5"
CLIENTSIZE=200


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

    with grpc.insecure_channel('localhost:50051', options=SETTINGS) as channel:
        stub = todo_pb2_grpc.FedMLStub(channel)
        msg=ClientgetWeight(stub)
        print('ClientgetWeight done')

        #TODO: Client do local training here


        msg=ClientsendWeight(stub) 
        print('ClientsendWeight done')

        # msg=ClientreCheck(stub)
        return msg
    



WORKER_NUMBER=2

if __name__ == '__main__':
    logging.basicConfig()
    for i in range(10):
        a=run()
        print('Round over')