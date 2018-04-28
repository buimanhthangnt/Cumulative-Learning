import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class CL_Regression:
    def __init__(self, input_data=None, labels=None, num_epochs=1000):
        self.learning_rate = 0.01
        self.num_epochs = num_epochs
        self.batch_size = 128
        self.input_dim = 6
        self.hidden_dim_1 = 20
        self.hidden_dim_2 = 20
        if input_data is not None and labels is not None:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                input_data, labels, test_size=0.2, shuffle=True)
        self.build_model()

    def build_model(self):
        self.data = tf.placeholder(tf.float32, shape=[None, self.input_dim])
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])
        hidden1 = tf.layers.dense(self.data, self.hidden_dim_1, use_bias=True,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
        hidden1 = tf.sigmoid(hidden1)
        hidden1 = tf.nn.dropout(hidden1, keep_prob=0.95)
        hidden2 = tf.layers.dense(hidden1, self.hidden_dim_2, use_bias=True,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
        hidden2 = tf.sigmoid(hidden2)
        hidden2 = tf.nn.dropout(hidden2, keep_prob=0.95)
        self.predicts = tf.layers.dense(hidden2, 1, use_bias=True,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.loss = tf.reduce_mean(tf.squared_difference(self.predicts, self.labels))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        best_test_loss = 99999
        is_stopped = False
        early_stop = 10
        count = 0

        for i in range(self.num_epochs):
            if is_stopped: break
            n_sample = self.X_train.shape[0]
            data_x, data_y = shuffle(self.X_train, self.y_train)
            batches = [(data_x[k:k + self.batch_size], data_y[k:k + self.batch_size])
                       for k in range(0, n_sample, self.batch_size)]
            losses = []

            for data, labels in batches:
                if (data.shape[0] != self.batch_size): continue
                _, loss = sess.run([self.optimizer, self.loss], feed_dict={self.data: data, self.labels: labels})
                losses.append(loss)

            if i % 200 == 0:
                train_loss = np.mean(np.array(losses))
                test_loss = sess.run(self.loss, feed_dict={self.data: self.X_test, self.labels: self.y_test})
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    count = 0
                    saver = tf.train.Saver()
                    saver.save(sess, "./regression_weight/regression.ckpt")
                else:
                    count += 1
                    print("Number of bad continuous process: %d" % count)
                    if count > early_stop:
                        is_stopped = True
                        break
                print("Epoch %i finished, train loss = %f, test loss = %f" % (i, train_loss, test_loss))
        print("Best test loss: %f\n\n" % best_test_loss)

    def generate_labels(self, data):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, "./regression_weight/regression.ckpt")
            y_pred = sess.run(self.predicts, feed_dict={self.data: data})
            with open("./gen_data/labels.txt", "w") as myfile:
                for i in y_pred:
                    myfile.write(str(i[0]) + "\n")

    def test(self, data, labels):
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, "./regression_weight/regression.ckpt")
        loss = sess.run(self.loss, feed_dict={self.data: data, self.labels: labels})
        print("Regression test, loss = %f" % loss)
