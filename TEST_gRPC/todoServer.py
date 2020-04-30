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
import time
import datetime

from models import ANN_model
model=ANN_model()
model.save_weights('server/ANN_model.h5') #TODO: hash this line if want the server to continue after crash..???


WEIGHT_PATH='server/ANN_model.h5'
EVALUATE_SAMPLE_FOLDER='9' ##TODO:??
EVALUATE_RESULT_PATH='server/evaluate'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.txt'


def get_nb_matrix(model_weights,nb): #used to support np.average
    model_w=[]
    for layer in model_weights:
        layer_w=np.empty(layer.shape)
        layer_w.fill(nb)
        model_w.append(layer_w)
    return model_w


def averageWeight(pathList,sizeList,model): 
    weight_list=[]
    reshapped_size_list=[]
    for i in range(len(pathList)):
        model.load_weights(pathList[i])
        weight_m = model.get_weights()
        weight_weight=get_nb_matrix(weight_m,sizeList[i])
        weight_list.append(weight_m)
        reshapped_size_list.append(weight_weight)
    
    avg_weight=np.average(weight_list, axis=0, weights=reshapped_size_list)

    print(sizeList)
    return avg_weight


class FedMLServicer(todo_pb2_grpc.FedMLServicer):

    def __init__(self):
        self.clientPathList=[]
        self.worker_nb=2         #TODO:??????
        self.isready=False
        self.sizeList=[]
        self.model=ANN_model()

    def evaluateModel(self):


        print('evaluation done~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        with open(EVALUATE_RESULT_PATH,'a') as t:
            acc=3
            t.write("{}\n".format(acc))

        pass


    def getWeight(self, request, context):

        with open(WEIGHT_PATH,'rb') as file:
            encoded_string = base64.b64encode(file.read())

        MAX_MESSAGE=100000
        step=len(encoded_string)/MAX_MESSAGE
        step=math.ceil(step) 
        for i in range(step):
            msg=encoded_string[i*MAX_MESSAGE:(i+1)*MAX_MESSAGE]
            yield todo_pb2.modelWeight(model=msg)

    
    
    def sendWeight(self, request, context):
        '''
        server collect and save the model from client
        '''

        for loops in range(100): #if previous round is not yet finished, hold for a while
            if self.isready==True:
                time.sleep(1)
            else:
                pass

        new_string=''
        for chunk in request:
            new_string=new_string+chunk.model
            current_CLIENT=chunk.clientname
            current_SIZE=chunk.clientsize

        current_PATH='server/'+current_CLIENT
        with open(current_PATH,'wb') as file:
            file.write(base64.b64decode(new_string))


        if current_PATH not in self.clientPathList:
            self.clientPathList.append(current_PATH)
            self.sizeList.append(current_SIZE)


        if len(self.clientPathList)==self.worker_nb:
            avg_weight=averageWeight(self.clientPathList,self.sizeList,self.model)  ## Here do the average
            print('average done~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            self.model.set_weights(avg_weight)
            self.model.save_weights(WEIGHT_PATH) 
            self.evaluateModel()    ## TODO: Here do the evaluation & write result to a txt file

            self.isready=True  #SET back these param for next epoch
            self.sizeList=[]


        myresponse = todo_pb2.myResponse()
        myresponse.value = 0

        for loops in range(100): ### wait till all client is ready
            if self.isready:
                myresponse.value = 1
                self.clientPathList.remove(current_PATH)
                if self.clientPathList==[]:
                    self.isready=False ### the last one to leave will reset self.isready for new round
                return myresponse
            else:
                time.sleep(1)
        return myresponse



    def reCheck(self, request, context): ### double check self.isready for new round  //  NOT USED
        myresponse = todo_pb2.myResponse(value = 2)
        if self.isready:
            if self.clientPathList==[]:
                self.isready=False
                myresponse = todo_pb2.myResponse(value = 3)
        if self.isready==False:
            if self.clientPathList!=[]:
                self.clientPathList=[]

        return myresponse




def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    todo_pb2_grpc.add_FedMLServicer_to_server(
        FedMLServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
