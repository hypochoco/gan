import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
from gan import generator_loss, discriminator_loss, generator_optimizer, discriminator_optimizer
from tqdm import tqdm

# --- training ---

def train():
    pass

if __name__ == "main": # ran as main file
    print(f"\n\n--- running training ---\n\n") # console message

    # TODO:
        # loading in a gan...
        # loading in a dataset... 
        # run he gan training... 

        # how do i structure this...
        # gan has a training function that takes all this information in...

        # gan class can be initialized with multiple types of gans and architectures...
        # naive gan...
        # minibatch gan... 





# --- visualizations ---

def generate_and_save_images(model: tf.keras.Sequential, epoch: int, show: bool=False, EDGE_SIZE=5):
    """ use a generator to make images from random noise """

    noise = tf.random.normal([EDGE_SIZE * EDGE_SIZE, 100]) # use noise rather than input
    # generated_image = generator(noise, training=False)

    generated_images = model([noise, tf.zeros((noise.shape[0], 2))], training=False)

    plt.figure(figsize=(EDGE_SIZE, EDGE_SIZE))
    plt.tight_layout()
    plt.axis('off')
    
    for i in range(EDGE_SIZE * EDGE_SIZE):
        ax = plt.subplot(EDGE_SIZE, EDGE_SIZE, i + 1)
        ax.set_axis_off()
        ax.imshow(generated_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')

    if show: # show if enabled
        plt.show()
    plt.savefig(f"outputs/image_at_epoch_{epoch:04d}.png")
    plt.close()

# --- train step ---

@tf.function
def train_step(images, labels, NOISE_DIM, BATCH_SIZE, generator, discriminator):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    # To make sure we know what is done, we will use a gradient tape instead of compiling
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Training the generator
        generated_images = generator([noise, labels] , training=True) 

        # Training the discriminator
        real_output = discriminator([images, labels], training=True)           # Training the discriminator on real images
        fake_output = discriminator([generated_images, labels], training=True) # Training the discriminator on fake images

        # Calculating the losses
        gen_loss =  generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        # Building the gradients
        gradients_of_generator =     gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        
        # Applying the gradients (backpropagation)
        generator_optimizer.apply_gradients(    zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_loss, disc_loss

# # Notice the use of `tf.function`
# # This annotation causes the function to be "compiled".
# @tf.function
# def train_step(images, noise_dim, BATCH_SIZE: int, generator: tf.keras.Sequential, discriminator: tf.keras.Sequential):
    
#     noise = tf.random.normal([BATCH_SIZE, noise_dim])

#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#         generated_images = generator(noise, training=True)

#         # TODO: add noise # noise added but what about the noise and the type of noise...
#         # TODO: figure out mini batch discrimination

#         real_output = discriminator(images, training=True)
#         fake_output = discriminator(generated_images, training=True)

#         gen_loss = generator_loss(fake_output)
#         disc_loss = discriminator_loss(real_output, fake_output)

#     gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
#     gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

#     generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
#     discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

#     return gen_loss, disc_loss

# --- train ---

def train(dataset, train_labels, BATCH_SIZE, generator, discriminator):

    epochs = 50
    noise_dim = 100

    checkpoint_dir = './outputs/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator
    )

    num_batches = int(dataset.shape[0]/BATCH_SIZE) # Amount of batches
    for epoch in range(epochs):
        start = time.time() # Timing the epoch

        pbar = tqdm(range(num_batches))

        sum_gen_loss, sum_disc_loss = 0, 0

        for batch_idx in pbar: # For each batch
            images = dataset[batch_idx*BATCH_SIZE : (batch_idx+1)*BATCH_SIZE]
            labels = train_labels[batch_idx*BATCH_SIZE : (batch_idx+1)*BATCH_SIZE]
            gen_loss, disc_loss = train_step(images, labels, noise_dim, BATCH_SIZE, generator, discriminator)
            sum_gen_loss += gen_loss
            sum_disc_loss += disc_loss
            pbar.set_description(f"gen_loss: {gen_loss}, disc_loss: {disc_loss}")

        generate_and_save_images(
            generator,
            epoch + 1
        )
        if (epoch + 1) % 5 == 0: # Save the model every 15 epochs
            checkpoint.save(file_prefix = checkpoint_prefix)

        # stats logging
        print (f"Time for epoch {epoch + 1} is {time.time()-start} sec, gen_loss: {sum_gen_loss}, disc_loss: {sum_disc_loss}")

    generate_and_save_images( # Generate after the final epoch
        generator,
        epochs
    )

    # EPOCHS = 50
    # noise_dim = 100

    # checkpoint_dir = './outputs/training_checkpoints'
    # checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    # checkpoint = tf.train.Checkpoint(
    #     generator_optimizer=generator_optimizer,
    #     discriminator_optimizer=discriminator_optimizer,
    #     generator=generator,
    #     discriminator=discriminator
    # )

    # for epoch in range(EPOCHS): # epocs

    #     start = time.time()

    #     sum_gen_loss, sum_disc_loss = 0, 0

    #     pbar = tqdm(dataset) # batches
    #     for image_batch in pbar:
    #         gen_loss, disc_loss = train_step(image_batch, train_labels, noise_dim, BATCH_SIZE, generator, discriminator)
    #         pbar.set_description(f"gen_loss: {gen_loss}, disc_loss: {disc_loss}")
    #         sum_gen_loss += gen_loss
    #         sum_disc_loss += disc_loss

    #     generate_and_save_images(
    #         generator,
    #         epoch + 1
    #     )
    #     if (epoch + 1) % 5 == 0: # Save the model every 15 epochs
    #         checkpoint.save(file_prefix = checkpoint_prefix)

    #     # stats logging
    #     print (f"Time for epoch {epoch + 1} is {time.time()-start} sec, gen_loss: {sum_gen_loss}, disc_loss: {sum_disc_loss}")

    # generate_and_save_images( # Generate after the final epoch
    #     generator,
    #     EPOCHS
    # )

def train_mnist(generator, discriminator):

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

    train_labels = tf.one_hot(train_labels, 2)
    train_labels = tf.reshape(train_labels, (train_labels.shape[0], 2))

    # training
    train(train_dataset, train_labels, BATCH_SIZE, generator, discriminator)