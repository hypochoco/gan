import tensorflow as tf
import numpy as np
import pandas as pd

# --- preprocess datasets ---

def preprocessMNIST():

    # load and format mnist data
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

    BUFFER_SIZE = 60000
    BATCH_SIZE = 256

    # Batch and shuffle the data
    # train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    train_dataset = train_images

    print(train_labels.shape)

    # TODO: wtf is this doing???
    train_labels = tf.one_hot(train_labels, 2)
    train_labels = tf.reshape(train_labels, (train_labels.shape[0], 2))

    return train_images, train_labels

def preprocessESL():

    # load and format mnist data
    train_df = pd.read_csv("datasets/sign-language/sign_mnist_train/sign_mnist_train.csv")
    test_df = pd.read_csv("datasets/sign-language/sign_mnist_test/sign_mnist_test.csv")

    

    x_train = train_df.values[:,0:-1]
    x_test = test_df.values[:,0:-1]

    x_train = x_train / 255
    x_test = x_test / 255

    x_train = x_train.reshape(-1,28,28,1)
    x_test = x_test.reshape(-1,28,28,1)

    # f, ax = plt.subplots(2,5) 
    # f.set_size_inches(10, 10)
    # k = 0
    # for i in range(2):
    #     for j in range(5):
    #         ax[i,j].imshow(x_train[k].reshape(28, 28) , cmap = "gray")
    #         k += 1
    #     plt.tight_layout()  

    # plt.show()

