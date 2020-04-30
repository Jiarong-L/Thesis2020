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



def node_training_process(index_path,shared_index,central_weight_path,local_epoch,batch_size=50,augment=False,local_iid=False):
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

        # load index
        index1 = np.load(index_path) 

        if shared_index!=[]:
            shared1 = np.array([])
            for x in shared_index:
                b=np.load(x)
                shared1 = np.concatenate((shared1, b))
            index1 = np.concatenate((index1, shared1))

        # load & processing data
        (x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data() 

        x_train_i=x_train[index1]
        y_train_i=y_train[index1]

        autotune=tf.data.experimental.AUTOTUNE
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
        save_dir = index_path[:-9]

        model=ANN_model()
        model.load_weights(central_weight_path)

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