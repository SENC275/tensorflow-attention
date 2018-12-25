import math
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.01
training_epochs = 10
batch_size = 128

TB_SUMMARY_DIR = './tb/mnist'

keep_prob = tf.placeholder(tf.float32)


class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name

        self._build_net()

    def _build_net(self):
        self.training = tf.placeholder(tf.bool)

        self.X = tf.placeholder(tf.float32, [None, 784])
        self.Y = tf.placeholder(tf.float32, [None, 10])
        self.X_image = tf.reshape(self.X, [-1, 28, 28, 1])
        tf.summary.image('input', self.X_image, 3)

        with tf.variable_scope('layer1') as scope:
            W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
            L1 = tf.nn.conv2d(self.X_image, W1, strides=[1,1,1,1], padding='SAME')
            L1 = tf.nn.relu(L1)

            attention_vec = tf.layers.dense(
                L1,
                32,
                activation=tf.nn.softmax,
            )

            L1 = tf.multiply(attention_vec, L1)

            tf.summary.histogram("X", self.X_image)
            tf.summary.histogram("weights", W1)
            tf.summary.histogram("layer", L1)

        with tf.variable_scope('layer2') as scope:
            W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
            L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
            L2 = tf.nn.relu(L2)
            L2 = tf.nn.max_pool(L2,
                                ksize=[1,2,2,1],
                                strides=[1,2,2,1],
                                padding='SAME')

            tf.summary.histogram("weights", W2)
            tf.summary.histogram("layer", L2)

        with tf.variable_scope('layer3') as scope:
            W3 = tf.Variable(tf.random_normal([3,3,64,128], stddev=0.01))
            L3 = tf.nn.conv2d(L2, W3, strides=[1,1,1,1], padding='SAME')
            L3 = tf.nn.relu(L3)
            L3 = tf.nn.max_pool(L3,
                                ksize=[1,2,2,1],
                                strides=[1,2,2,1],
                                padding='SAME')

            tf.summary.histogram("weights", W3)
            tf.summary.histogram("layer", L3)

        with tf.variable_scope('layer4') as scope:
            L3_flat = tf.reshape(L3, [-1, 7*7*128])
            W5 = tf.Variable(tf.random_normal([7*7*128, 10]))
            b5 = tf.Variable(tf.random_normal([10]))
            self.logits = tf.matmul(L3_flat, W5) + b5

            tf.summary.histogram("layer", self.logits)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y
        ))

        tf.summary.scalar("loss", self.cost)

        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(self.cost)

        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1)
        )

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def predict(self, x_test, training=False):
        return self.sess.run(self.logits,
                             feed_dict={self.X: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy,
                             feed_dict={self.X: x_test,
                                        self.Y: y_test,
                                        self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost,self.optimizer, tf.summary.merge_all()],
                             feed_dict={self.X: x_data,
                                        self.Y: y_data,
                                        self.training: training})

sess = tf.Session()
m1 = Model(sess,  "m1")
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter(TB_SUMMARY_DIR)
writer.add_graph(sess.graph)
global_step = 0

for epoch in range(10):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        cost, _, summary = m1.train(batch_xs, batch_ys)
        writer.add_summary(summary, global_step=global_step)
        avg_cost += cost / total_batch
        global_step += 1


    print("Epoch:", "%04d" % (epoch + 1), "cost =", "{:.9f}".format(avg_cost))
    print("Accuracy: ", m1.get_accuracy(mnist.test.images, mnist.test.labels))
