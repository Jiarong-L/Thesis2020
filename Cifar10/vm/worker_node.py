import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import glob

from models import ANN_model


def img_augument(img,lab):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_hue(img,0.1)
    img = tf.image.random_contrast(img, 0.5, 2)
    return img, lab


def set_iid(y_train,x_train):
    category_index=[]
    for i in range(10):
        index0 = np.where(y_train == i)
        index = index0[0]
        np.random.shuffle(index)
        category_index.append(index)
    
    min_cato=1000000

    for x in category_index:
        if min_cato>x.shape[0]:
            min_cato=x.shape[0]

    if min_cato==0:
        return y_train,x_train

    cat_i=np.concatenate([x[:min_cato] for x in category_index])

    return y_train[cat_i],x_train[cat_i]



def node_training_process(index_path,shared_index,central_weight_path,local_epoch,batch_size=50,augment=False,local_iid=False,node_evl=False):
    '''
    1. Get index and initial_weights from central,
    2. Load & prepare the dataset accordingly,
    3. Training,
    4. Return weights to central
    * In reality, it doesn't need index from central, it can read all local data like glob.glob('data_dir')
    * Saving node weights locally can be a safe way if node have many data, but here we just neglect this
    '''
    g = tf.Graph()
    with g.as_default(): #tf.graph to solve memory leak

        # load & processing data
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data() 

        autotune=tf.data.experimental.AUTOTUNE

        # load index
        index1 = np.load(index_path) 

        # assign node_evl_set (1/2)
        if node_evl:
            evl_p = index_path[:-9]+'evl_index.npy'
            evl_index = np.load(evl_p)
            x_test_n=x_test[evl_index]
            y_test_n=y_test[evl_index]
            node_evl_list = []
            total_node_evl_list = []
            for i in range(10):
                index0 = np.where(y_test_n == i)
                index = index0[0]
                x_evl=tf.data.Dataset.from_tensor_slices(x_test_n[index])
                y_evl=tf.data.Dataset.from_tensor_slices(y_test_n[index])
                node_evl_set = tf.data.Dataset.zip((x_evl, y_evl))
                node_evl_set = node_evl_set.repeat().batch(1).prefetch(buffer_size=autotune)
                total_node_evl = len(index)
                node_evl_list.append(node_evl_set)
                total_node_evl_list.append(total_node_evl)


        if shared_index!=[]:
            shared_test_index = np.array([0])
            for x in shared_index:
                b=np.load(x)
                index1 = np.concatenate((index1, b))
                shared_test_index = np.concatenate((shared_test_index, b))
            shared_test_index = shared_test_index[1:]
            x_test_shared=x_train[shared_test_index]
            y_test_shared=y_train[shared_test_index]
            x_shared_evl=tf.data.Dataset.from_tensor_slices(x_test_shared)
            y_shared_evl=tf.data.Dataset.from_tensor_slices(y_test_shared)
            shared_evl_set = tf.data.Dataset.zip((x_shared_evl, y_shared_evl))
            shared_evl_set = shared_evl_set.repeat().batch(batch_size).prefetch(buffer_size=autotune)
            total_shared_evl = shared_test_index.shape[0] ###################################


        x_train_i=x_train[index1]
        y_train_i=y_train[index1]

        buffer_size = x_train_i.shape[0]
        total_traning=index1.shape[0]

        x_tr=tf.data.Dataset.from_tensor_slices(x_train_i)
        y_tr=tf.data.Dataset.from_tensor_slices(y_train_i)

        if local_iid==True:
            y_train_i2 , x_train_i2 =set_iid(y_train_i,x_train_i)
            x_tr=tf.data.Dataset.from_tensor_slices(x_train_i2)
            y_tr=tf.data.Dataset.from_tensor_slices(y_train_i2)

        train_set=tf.data.Dataset.zip((x_tr, y_tr))
        if augment==True:
            train_set=train_set.map(img_augument).shuffle(buffer_size,reshuffle_each_iteration=True).repeat().batch(batch_size).prefetch(buffer_size=autotune)
        else:
            train_set=train_set.shuffle(buffer_size,reshuffle_each_iteration=True).repeat().batch(batch_size).prefetch(buffer_size=autotune)


    # Training & save
        # THIS LINE SHOULD BE THE FIRST
        save_dir = index_path[:-9]

        model=ANN_model()
        model.load_weights(central_weight_path)


        # node_evl before training (2/2)
        if node_evl:
            filename = os.path.join(save_dir,'node_EVAL_before_training.txt')
            with open(filename,'a') as file_handle:
                for i in range(10):
                    if total_node_evl_list[i]==0:
                        file_handle.write('200')
                        file_handle.write(' ')
                    else:
                        [loss, acc] = model.evaluate(node_evl_list[i],steps=total_node_evl_list[i]//1,verbose=0)
                        file_handle.write(str(acc))
                        file_handle.write(' ')
                file_handle.write('\n')


        # see if overtrained over the shared index
        if shared_index!=[]:
            [loss, acc]=model.evaluate(shared_evl_set,steps=total_shared_evl//batch_size,verbose=0)
            filename = os.path.join(save_dir,'shared_EVAL.txt')
            with open(filename,'a') as file_handle:
                    file_handle.write(str(loss))
                    file_handle.write(' ')
                    file_handle.write(str(acc))
                    file_handle.write('\n')


        # test the loaded model to see if it's overtrainned? mention it's last epo's acc
        [self_loss, self_acc]=model.evaluate(train_set,steps=total_traning//batch_size,verbose=0)
        filename = os.path.join(save_dir,'self_EVAL.txt')
        with open(filename,'a') as file_handle:
                file_handle.write(str(self_loss))
                file_handle.write(' ')
                file_handle.write(str(self_acc))
                file_handle.write('\n')


        history = model.fit(train_set,
                            epochs=local_epoch,
                            steps_per_epoch=total_traning//batch_size,
                            verbose=0)

        # return model_weight   
        model_weights=model.get_weights() 

    del model

    # TODO: Change/add validation based on the split of data on the worker node -----------> save locally in worker nodes.
    #       And compare this weighted average to current one (a centralized testing set)


    return model_weights,total_traning

# index_path