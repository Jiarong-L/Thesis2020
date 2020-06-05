# cifar10

Using python3 and tf2.0.1 cpu/gpu version

## The Simulation.ipython will:
1. Assign data to worker_node (Save dataset's index to worker's folder)  
2. Build and compile the model, save the initial model weights  


## The central_node will:
1. Ask worker_node to start local training
2. Collects and weighted average the model weights
3. Save model weights
4. Evaluate on testing set % save evaluate results
5. Iter previous acts

## The Ploting.ipython will:
1. Plot result in the 'result' folder


## The worker_node will:
1. Build and compile the model (same as central)  
2. Load model weights from central_node  
3. Load data (Load the assigned index, then prepare the data accordingly)  
4. Start local training  
5. Save model weights locally  
6. Evaluate on testing set  
7. Return node data amount, model weights to central_node  


## This example is a simulation on a single machine.  
Using the cifar10 dataset: 50000 training set + 10000 testing set. 10 classes.  
Split the testing set, but central_node owns the whole testing set.  
*assume each folder is a machine

## Record of each test
Link: https://pan.baidu.com/s/1VBTusDNyJN9DtnzP9QBZlg  
key: bcvj