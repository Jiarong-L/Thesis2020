# A simple grpc system  
In this system assume we have 3 client and a server. For convinence's sake, here is the local demo which is successfully tested on the cloud environment.  
Here's an example:  
Images stored in 'preprocessed' folder. Client001 read in 'index/worker1.pkl' and load its training data accordingly. Client001 load weight from server/ do training and save it to 'client001/client_weights'.   
Server get weight from Client001 and save it to 'server/client001/client_weights', then merge the model and save it to 'server/server_weight.h5'.

# What to be trained:
The federated task of [DR](https://github.com/Jiarong000/Thesis2020/tree/master/DR)  
Model to str method refers to [Mayank Shah](https://github.com/mayankshah1607/federated-learning-with-grpc-docker/blob/master/node/functions.py)


## Install dependencys
python -m pip install --upgrade pip  
python -m pip install grpcio  
python -m pip install grpc_tools  

## Run proto file
python3 -m grpc_tools.protoc -I=. --python_out=. --grpc_python_out=. ./todo.proto  
 

[get 'todo_pb2.py' & 'todo_pb2_grpc.py']

## System Description
### Message Type
    message modelWeight {
        string model = 1;
        string clientname = 2;
        int32 clientsize=3;
    }

    message myResponse {
        int32 value = 1;
    }

    message void {}  

### Service:
    rpc getWeight(void) returns (stream modelWeight);
    rpc sendWeight(stream modelWeight) returns (myResponse);

### What happened  
#### Server   
1. start Server : (python3 todoServer.py)    
2. Server initialize model & weights, it will have 3 client for each total round    
--round start--
3. Receive the void request from client, read in and return model weight to it. (getWeight function)  
4. Receive the first client's weight, record each client's name (assume name is the path to save) and save client's returned weights to specific folder. Wait until all 3 clients finished, merge and evaluate the model. Send 'myResponse' back to clients. ('sendWeight' function)  
5. NEXT ROUND  

#### Client
1. connection to server: grpc.insecure_channel('SERVER_IP:50051', options=SETTINGS)  
2. Get weights from server and save locally(ClientgetWeight function)  
3. Create the model, load the weight and do training. Save trained model and training history locally. 
4. Upload weight and name and node size to server, wait for 'myResponse' from the server. (ClientsendWeight function)
5. NEXT ROUND
