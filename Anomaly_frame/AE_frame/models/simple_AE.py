import tensorflow as tf
import tensorflow.contrib.slim as slim

class autoencoder():
    def encoder(x, z_dim, reuse=False):
        with tf.variable_scope("enc"):
            # image is 256 x 256 x input_c_dim
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            with slim.arg_scope([slim.conv2d], padding='SAME', activation_fn=tf.nn.leaky_relu, weights_initializer = tf.initializers.he_normal()):
                net = slim.conv2d(inputs=x, num_outputs=64, kernel_size=[3, 3], stride=2, normalizer_fn=slim.batch_norm, scope='conv1') # (128, 128, 64)
                net = slim.conv2d(inputs=net, num_outputs=128, kernel_size=[3, 3], stride=2, normalizer_fn=slim.batch_norm, scope='conv2') # (64, 64, 128)
                net = slim.conv2d(inputs=net, num_outputs=256, kernel_size=[3, 3], stride=2, normalizer_fn=slim.batch_norm, scope='conv3') # (32, 32, 256)
                net = slim.flatten(net, scope='flatten')
                net = slim.fully_connected(net, z_dim)
                net = tf.expand_dims(net, axis=2)
            return net

    def decoder(z, reuse=False):
        with tf.variable_scope("dec"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected], activation_fn=tf.nn.leaky_relu, weights_initializer = tf.initializers.he_normal()):
                net = slim.fully_connected(z, 32*32*256)
                net = tf.reshape(net, [-1,32,32,256]) # (32, 32, 256)
                net = slim.conv2d_transpose(inputs=net, num_outputs=128, kernel_size=[3, 3], stride=2) # (64, 64, 128)
                net = slim.conv2d_transpose(inputs=net, num_outputs=64, kernel_size=[3, 3], stride=2) # (128, 128, 64)
                net = slim.conv2d_transpose(inputs=net, num_outputs=1, kernel_size=[3, 3], stride=2)  # (256, 256, 1)
            return net