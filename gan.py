import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input
from keras.datasets import mnist
from tqdm import tqdm
from keras.layers.activation import LeakyReLU
import tensorflow as tf

def build_generator():
    generator= Sequential() # initialize model

    generator.add(Dense(units=256, input_dim=100)) # input layer
    generator.add(LeakyReLU(0.2)) # activation with leakyrelu

    generator.add(Dense(units=512)) # batch normalization
    generator.add(Dense(units=1024))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(units=784, activation='tanh')) # output layer with 784 (28x28)

    generator.compile( # compiling the generator
        loss='binary_crossentropy', 
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    )

    return generator

def build_discriminator():
    discriminator=Sequential() # initialize model

    discriminator.add(Dense(units=1024,input_dim=784)) # input layer
    discriminator.add(LeakyReLU(0.2)) # activation with leakyrelu
    
    discriminator.add(Dropout(0.2)) # dropout to reduce overfitting
    
    discriminator.add(Dense(units=512)) # second layer
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    
    discriminator.add(Dense(units=256)) # third layer
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
  
    discriminator.add(Dense(units=128)) # forth layer
    discriminator.add(LeakyReLU(0.2))
  
    discriminator.add(Dense(units=1,activation='sigmoid')) # output with sigmoid
  
    discriminator.compile( # compiling disciminator
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002,beta_1=0.5)
    )
  
    return discriminator

def gan_net(generator: Sequential, discriminator: Sequential):
    
    discriminator.trainable = False # freeze weights
    
    inp = Input(shape=(100,)) # keras tensor of shape 100
    
    X = generator(inp)
    out = discriminator(X)
    
    gan = Model(inputs=inp, outputs=out)

    gan.compile(
        loss='binary_crossentropy',
        optimizer='adam'
    )

    return gan

def plot_images(epoch, generator: Sequential, dim=(10,10), figsize=(10,10)):

    noise=np.random.normal(loc=0,scale=1,size=[100,100]) # normally distributed noise (100x100)
    generated_images = generator.predict(noise) # generate image from noise
  
    # reshape the generated image
    generated_images = generated_images.reshape(100,28,28)
  
    # plo image
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        
        plt.subplot(dim[0],dim[1],i+1)
        plt.imshow(generated_images[i],cmap='gray',interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()

def train(X_train, epochs=5, batch_size=128):
    #initializing the GAN
    generator = build_generator()
    discriminator = build_discriminator()
    gan=gan_net(generator, discriminator)
    
    #training the model for specified epochs
    for epoch in range (1, epochs+1):
        print("###### @ Epoch",epoch)
        #tqdm module helps to generate a status bar for training
        for _ in tqdm(range(batch_size)):
            #random noise with size batch_sizex100
            noise=np.random.normal(0,1,[batch_size,100])
            #generating images from noise
            generated_images=generator.predict(noise)
            #taking random images from the training
            image_batch=X_train[np.random.randint(low=0,high=X_train.shape[0]
                                                 ,size=batch_size)]
            #creating a new training set with real and fake images
            
      
            X=np.concatenate([image_batch,generated_images])
      
            #labels for generated and real data
            y_dis=np.zeros(2*batch_size)
            #label for real images
            y_dis[:batch_size]=1.0
      
            #training the discrminator with real and generated images
            discriminator.trainable=True
            discriminator.train_on_batch(X,y_dis)
      
            #labelling the generated images a sreal images(1) to trick the discriminator
      
            noise=np.random.normal(0,1,[batch_size,100])
            y_gen=np.ones(batch_size)
      
            #freezing the weights of the discriminant or while training generator
      
            discriminator.trainable=False
      
            #training the gan network
      
            gan.train_on_batch(noise,y_gen)
      
            #plotting the images for every 10 epoch
            if epoch==1 or epoch %10==0:
                
                
                plot_images(epoch,generator,dim=(10,10),figsize=(15,15))