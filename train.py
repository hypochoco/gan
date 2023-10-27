from gan import train
import numpy as np
import tensorflow as tf

#Unpacking the training data from mnist dataset
(X_train,_),(_,_)=tf.keras.datasets.mnist.load_data()

#converting to float type and normalizing the data

X_train=(X_train.astype(np.float32)-127.5)/127.5

#convert shape of X_train from (60000,28,28) to (60000, 784) -784 coloumns per row

X_train=X_train.reshape(60000,784)

train(X_train,epochs=5,batch_size=128)