import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt

# --- naive GAN ---




class NaiveGAN():

    generator: tf.keras.Sequential
    discriminator: tf.keras.Sequential

    def build_generator(self):
        pass
    
    def build_discriminator(self):
        pass

    def __init__(self):
        # make generator and the discriminator...

        pass

    def load_weights(self):
        pass




# --- minibatch GAN ---







# --- minibatch discrimination ---

class MinibatchDiscrimination(tf.keras.layers.Layer):

    def __init__(self, num_kernel, dim_kernel,kernel_initializer='glorot_uniform', **kwargs):
        self.num_kernel = num_kernel
        self.dim_kernel = dim_kernel
        self.kernel_initializer = kernel_initializer
        super(MinibatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(
            name='kernel', 
            shape=(input_shape[1], self.num_kernel*self.dim_kernel),
            initializer=self.kernel_initializer,
            trainable=True
        )
        super(MinibatchDiscrimination, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        activation = tf.matmul(x, self.kernel)
        activation = tf.reshape(activation, shape=(-1, self.num_kernel, self.dim_kernel))
        #Mi
        tmp1 = tf.expand_dims(activation, 3)
        #Mj
        tmp2 = tf.transpose(activation, perm=[1, 2, 0])
        tmp2 = tf.expand_dims(tmp2, 0)
        
        diff = tmp1 - tmp2
        
        l1 = tf.reduce_sum(tf.math.abs(diff), axis=2)
        features = tf.reduce_sum(tf.math.exp(-l1), axis=2)
        return tf.concat([x, features], axis=1)        
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + self.num_kernel)

# --- generator and discriminator ---

def build_generator():
    """ build generator, input is a 100 shaped noise vector """

    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

KERNEL_SIZE =    (5, 5)   # Kernel size for the convolutional layers
MOMENTUM =       0.9      # Momentum for the batch normalization layers
DROPOUT =        0.4      # Dropout rate
ALPHA =          0.2      # Alpha for the leaky ReLU slope

def generator_model(verbose=False):

    noise_input = layers.Input(shape=(100,)) # Noise input
    gen_cond_in = layers.Input(shape=(2,)) 
    
    # Constrain the generator with a condition
    merged_in = layers.Concatenate()([noise_input, gen_cond_in])

    hid = layers.Dense(256 * 7 * 7)(merged_in)
    hid = layers.Reshape((7, 7, 256))(hid)
    hid = layers.ReLU()(hid) 
    hid = layers.BatchNormalization(momentum=MOMENTUM)(hid)
    
    # 7 ==> 14
    hid = layers.Conv2DTranspose(   256, kernel_size=KERNEL_SIZE, strides=(2, 2), padding="same", use_bias=False)(hid)
    hid = layers.ReLU()(hid)
    hid = layers.BatchNormalization(momentum=MOMENTUM)(hid)
    
    # 14 ==> 28
    hid = layers.Conv2DTranspose(   128, kernel_size=KERNEL_SIZE, strides=(2, 2), padding="same", use_bias=False)(hid)
    hid = layers.ReLU()(hid)
    hid = layers.BatchNormalization(momentum=MOMENTUM)(hid)
    
    hid = layers.Conv2D(1, kernel_size=KERNEL_SIZE, strides=(1, 1), padding="same")(hid)
    out = layers.Activation("tanh")(hid)

    model = tf.keras.Model(inputs=[noise_input, gen_cond_in], outputs=out)
    
    if verbose:
        model.summary()
    
    return model

def discriminator_model(verbose=False):

    img_input = layers.Input(shape=(28, 28, 1)) # Image input
    disc_cond_in = layers.Input(shape=(2,)) # Condition input

    # 32 ==> 16
    hid = layers.Conv2D(64, kernel_size=KERNEL_SIZE, strides=(2, 2), padding='same', use_bias=False)(img_input) 
    hid = layers.LeakyReLU(alpha=ALPHA)(hid) 
    hid = layers.BatchNormalization(momentum=MOMENTUM)(hid)

    # 16 ==> 8
    hid = layers.Conv2D(128, kernel_size=KERNEL_SIZE, strides=(2, 2), padding='same', use_bias=False)(hid)
    hid = layers.LeakyReLU(alpha=ALPHA)(hid) 
    hid = layers.BatchNormalization(momentum=MOMENTUM)(hid)

    # 8 ==> 4
    hid = layers.Conv2D(256, kernel_size=KERNEL_SIZE, strides=(2, 2), padding='same', use_bias=False)(hid)
    hid = layers.LeakyReLU(alpha=ALPHA)(hid) 
    hid = layers.BatchNormalization(momentum=MOMENTUM)(hid)

    hid = layers.Flatten()(hid)
    
    hid =    layers.Dense(256)(hid)
    hid =    layers.LeakyReLU(alpha=ALPHA)(hid)
    
    # Indicating the discriminator the condition
    merged = layers.Concatenate()([hid, disc_cond_in])

    hid =    layers.Dense(256)(merged)
    hid =    layers.LeakyReLU(alpha=ALPHA)(hid)
    
    hid = MinibatchDiscrimination(num_kernel=100, dim_kernel=5, name="mbd")(hid)

    hid =    layers.Dense(256)(hid)
    hid =    layers.LeakyReLU(alpha=ALPHA)(hid)
    out = layers.Dense(1)(hid) # No sigmoid activation because we use Cross Entropy with from_logits=True
    
    model = tf.keras.Model(inputs=[img_input, disc_cond_in], outputs=out)
    
    if verbose:
        model.summary()
    
    return model

def build_discriminator():
    """ build discriminator, input is a 28x28x1 image """

    model = tf.keras.Sequential()

    model.add(layers.GaussianNoise(0.25) )

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# --- loss functions ---

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# --- optimizers ---

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)