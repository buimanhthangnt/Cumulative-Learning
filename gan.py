from __future__ import division, print_function, absolute_import

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle


class CL_Gan:
    def __init__(self, input_data=None):
        units = input_data.shape[0] // 24
        self.input_data = input_data
        self.learning_rate = 0.0005
        self.batch_size = 128
        self.num_epochs = 24000
        self.gen_hidden_dim = units
        self.disc_hidden_dim = units
        self.noise_dim = 3
        self.input_dim = 6
        self.weights = {
            'gen_hidden1': tf.Variable(self.xavier_init([self.noise_dim, self.gen_hidden_dim])),
            'gen_out': tf.Variable(self.xavier_init([self.gen_hidden_dim, self.input_dim])),
            'disc_hidden1': tf.Variable(self.xavier_init([self.input_dim, self.disc_hidden_dim])),
            'disc_out': tf.Variable(self.xavier_init([self.disc_hidden_dim, 1])),
        }
        self.biases = {
            'gen_hidden1': tf.Variable(tf.zeros([self.gen_hidden_dim])),
            'gen_out': tf.Variable(tf.zeros([self.input_dim])),
            'disc_hidden1': tf.Variable(tf.zeros([self.disc_hidden_dim])),
            'disc_out': tf.Variable(tf.zeros([1])),
        }
        self.build_model()

    def xavier_init(self, shape):
        return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

    def generator(self, x):
        hidden_layer = tf.matmul(x, self.weights['gen_hidden1'])
        hidden_layer = tf.add(hidden_layer, self.biases['gen_hidden1'])
        hidden_layer = tf.sigmoid(hidden_layer)
        out_layer = tf.matmul(hidden_layer, self.weights['gen_out'])
        out_layer = tf.add(out_layer, self.biases['gen_out'])
        return out_layer

    def discriminator(self, x):
        hidden_layer = tf.matmul(x, self.weights['disc_hidden1'])
        hidden_layer = tf.add(hidden_layer, self.biases['disc_hidden1'])
        hidden_layer = tf.nn.sigmoid(hidden_layer)
        out_layer = tf.matmul(hidden_layer, self.weights['disc_out'])
        out_layer = tf.add(out_layer, self.biases['disc_out'])
        out_layer = tf.nn.sigmoid(out_layer)
        return out_layer

    def build_model(self):
        self.gen_input = tf.placeholder(tf.float32, shape=[None, self.noise_dim])
        self.disc_input = tf.placeholder(tf.float32, shape=[None, self.input_dim])
        self.gen_sample = self.generator(self.gen_input)
        disc_real = self.discriminator(self.disc_input)
        disc_fake = self.discriminator(self.gen_sample)
        self.gen_loss = -tf.reduce_mean(tf.log(disc_fake))
        self.disc_loss = -tf.reduce_mean(0.5 * (tf.log(disc_real) + tf.log(1. - disc_fake)))
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        optimizer_disc = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gen_vars = [self.weights['gen_hidden1'], self.weights['gen_out'],
                    self.biases['gen_hidden1'], self.biases['gen_out']]
        disc_vars = [self.weights['disc_hidden1'], self.weights['disc_out'],
                     self.biases['disc_hidden1'], self.biases['disc_out']]
        self.train_gen = optimizer_gen.minimize(self.gen_loss, var_list=gen_vars)
        self.train_disc = optimizer_disc.minimize(self.disc_loss, var_list=disc_vars)

    def train(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        print("\n\nGAN is training")
        for i in range(1, self.num_epochs + 1):
            input_data = shuffle(self.input_data)
            batches = [input_data[k:k + self.batch_size]
                       for k in range(0, input_data.shape[0], self.batch_size)]
            for idx, batch in enumerate(batches):
                if batch.shape[0] != self.batch_size: continue
                z = np.random.uniform(-1., 1., size=[self.batch_size, self.noise_dim])

                feed_dict = {self.disc_input: batch, self.gen_input: z}
                _, _, gl, dl = sess.run([self.train_gen, self.train_disc, self.gen_loss, self.disc_loss],
                                        feed_dict=feed_dict)
                sess.run(self.train_gen, feed_dict=feed_dict)

                if (i % 200 == 0 or i == 1) and idx == 0:
                    print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

        saver = tf.train.Saver()
        saver.save(sess, "./gan_weight/gan.ckpt")

    def generate_data(self, size=None):
        if size is None and self.input_data is not None:
            size = self.input_data.shape[0]
        if size is None:
            print("size is None !!!")
            return
        z = np.random.uniform(-1., 1., size=[size, self.noise_dim])
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, "./gan_weight/gan.ckpt")
        g = sess.run(self.gen_sample, feed_dict={self.gen_input: z})
        with open("./gen_data/data.txt", "w") as myfile:
            for sample in g:
                st = str(sample[0])
                for v in sample[1:]:
                    st += "\t" + str(v)
                st += "\n"
                myfile.write(st)
