import argparse
import os
import tensorflow as tf
from models.simple_AE import autoencoder as autoencoder
import os

tf.set_random_seed(20)

def mse_criterion(input, target):
    return tf.square(input-target)

def l2_loss(input, recon):
    return tf.reduce_mean(tf.reduce_mean(mse_criterion(input, recon), axis=[1,2,3]))


######################################################################
###                         Set Parameters                         ###
######################################################################

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Parameters
train_data_path = 'D:\\'
n_epochs = 100
batch_size = 1
learning_rate = 0.0001
beta1 = 0.5
z_dim = 128


######################################################################
###                 Define Network (AE, VAE, AAE)                  ###
######################################################################

x = tf.placeholder(tf.float32, [None, 256, 256, 1], name='input')

# AE
z = autoencoder.encoder(x, z_dim)
x_r = autoencoder.decoder(z)

## VAE
#z = encoder(x)
## sampling 
## z = ...
#x_r = decoder(z)

######################################################################
###                 Define Loss & Optimizer Etc..                  ###
######################################################################

# Define loss function
loss = l2_loss(x, x_r)

# Optimizer
all_vars = tf.trainable_variables()
enc_vars = [var for var in all_vars if 'enc' in var.name]
enc_vars = [var for var in all_vars if 'dec' in var.name]
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(loss, var_list=all_vars)

# Tensorboard visualization
tf.summary.scalar(name='Autoencoder Loss', tensor=autoencoder_loss)
tf.summary.histogram(name='Real Distribution', values=real_distribution)
tf.summary.image(name='Generated Image', tensor=generated_images, max_outputs=10)
summary_op = tf.summary.merge_all()


######################################################################
###                        Start Training                          ###
######################################################################

saver = tf.train.Saver()
with tf.Session(config) as sess:
    writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)
    # Start Train
    for e in range(1, n_epochs+1):
        # Train minibatchs
        for b in range(n_batches):
            # load batch
            batch = load_batch(imgs, b)
            fetches = {
                'optimizer':optimizer,
                }
            sess.run(fetches, feed_dict={x : batch})
            
            # Write Summary & Log (if needed)
            if b % 50 == 0:
                fetches = {
                    'loss_':loss,
                    'input_image':x,
                    'latent_vector':z,
                    'recon_image':x_r,
                    'summary':summary_op,
                    }
                out = sess.run(fetches)
                loss_ = out['loss']
                summary = out['summary']
                # Write
                writer.add_summary(summary, global_step = step)
                with open(log_path + '/log.txt', 'a') as log:
                    log.write("Epoch: {}, iteration: {}\n".format(i, b))
                    log.write("Autoencoder Loss: {}\n".format(loss_))
        
        # Validation
        for b in range(n_vali_batches):
            # load batch
            batch = load_batch(vali_imgs, b)
            fetches = {
                'input_image':x,
                'latent_vector':z,
                'recon_image':x_r,
                'summary':summary_op,
                }
            out = sess.run(fetches, feed_dict={x : batch})

            # Write Summary & Log
            writer.add_summary(summary, global_step = step)
            with open(log_path + '/log.txt', 'a') as log:
                log.write("Epoch: {}, iteration: {}\n".format(i, b))
                log.write("Autoencoder Loss: {}\n".format(loss_))

        # Save Weights
        saver.save(sess, save_path=saved_model_path, global_step=step)