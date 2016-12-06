#!/usr/bin/python
# -*- coding: utf-8 *-*

import datetime
import os
import math
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

def progressbar(cur,total):
    percent = '{:.2%}'.format( float(cur)/total)
    sys.stdout.write('\r')
    sys.stdout.write('[%-50s] %s' % ( '=' * int(math.floor(cur * 50 /total)),percent))
    sys.stdout.flush()
    if cur == total:
        sys.stdout.write('\n')

class RecurrentNeuralNetwork:
    def __init__(self):
        self.session = tf.Session()
        self.inputs = None
        self.input_layer = None
        self.label_layer = None
        self.weights = None
        self.biases = None
        self.lstm_cell = None
        self.prediction = None
        self.loss = None
        self.trainer = None

    def __del__(self):
        self.session.close()

    def train(self, train_x, train_y, learning_rate=0.01, epochs=1, batch_n=1, input_n=1):
        seq_n = len(train_x)
        input_n = len(train_x[0])
        output_n = len(train_y[0])

        # self.input_layer = tf.placeholder(tf.float32, in_shape)
        self.inputs = tf.placeholder(tf.float32, [batch_n, input_n])
        self.label_layer = tf.placeholder(tf.float32, [output_n])
        self.input_layer = [tf.reshape(i, (1, input_n)) for i in tf.split(0, batch_n, self.inputs)]

        self.weights = tf.Variable(tf.random_normal([input_n, output_n]))
        self.biases = tf.Variable(tf.random_normal([output_n]))
        self.prediction = tf.matmul(self.inputs, self.weights) + self.biases
        #self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.prediction, self.label_layer))
        #self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        self.loss = tf.reduce_mean(tf.square(self.prediction - self.label_layer))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)


        initer = tf.global_variables_initializer()
        writer = tf.train.SummaryWriter("./graph", self.session.graph)

        tf.scalar_summary("loss", self.loss)
        #tf.scalar_summary("prediction", self.prediction[0][0])
        merged_summary = tf.merge_all_summaries()

        self.session.run(initer)
        for epoch in range(epochs):
            for idx in range(seq_n):
                input_x = train_x[idx:idx+1]
                output_y = train_y[idx]
                feed_dict = {self.inputs: input_x, self.label_layer: output_y}
                _, summary = self.session.run([self.trainer, merged_summary], feed_dict=feed_dict)

                if False:
                    cur_prediction = self.session.run(self.prediction, feed_dict=feed_dict)
                    cur_loss = self.session.run(self.loss, feed_dict=feed_dict)
                    print (cur_prediction[0][0], cur_loss)
                
                writer.add_summary(summary, idx)

    def predict(self, test_x, test_y, batch_n):
        seq_n = len(test_x)
        input_n = len(test_x[0])

        acc_predict_cnt = 0
        acc_cnt = 0
        no_acc_predict_cnt = 0
        no_acc_cnt = 0
        for idx in range(seq_n):
            input_x = test_x[idx:idx + batch_n]
            label_y = test_y[idx]
            predict_y = self.session.run(self.prediction, feed_dict={self.inputs: input_x})
            #print("line %d:%f %f" % (idx, label_y, predict_y))
            if label_y >= 1.0:
                acc_cnt += 1
                if label_y == int(predict_y+0.5):
                    acc_predict_cnt += 1
            else:
                no_acc_cnt += 1
                if label_y == int(predict_y+0.5):
                    no_acc_predict_cnt += 1


        # 有事故，预测成功的准确率
        acc_accuracy = float(acc_predict_cnt)/acc_cnt
        no_acc_accuracy = float(no_acc_predict_cnt)/no_acc_cnt

        # 无事故，预测成功的准确率
        print("no_acc_predict_cnt=%d, acc_predict_cnt=%d"%(no_acc_cnt, acc_cnt))
        print("predict no_acc_predict_cnt=%d, acc_predict_cnt=%d"%(no_acc_predict_cnt, acc_predict_cnt))
        print("acc accuracy= %f"% acc_accuracy)
        print("no acc accuracy= %f"% no_acc_accuracy)

    def test(self, train_x, train_y, test_x, test_y, batch_n, epochs):
        self.train(train_x, train_y, batch_n=batch_n, epochs=epochs)
        self.predict(test_x, test_y, batch_n=batch_n)


def normalize(x):
    return (x-min(x))/(max(x)-min(x))

def data_import(file, delimiter=','):
    x_cols = (1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    y_cols = (0)
    
    x = np.genfromtxt(file, delimiter=delimiter, skip_header=True, usecols=x_cols)
    # visibility 4
    x[:,3] = normalize(x[:,3])
    # wind 5
    x[:,4] = normalize(x[:,4])
    # wind_dir, ignore
    # x[:,5] = normalize(x[:,5])

    y = np.genfromtxt(file, delimiter=delimiter, skip_header=True, usecols=y_cols)
    y = np.array([[value] for value in y])
    return x, y

if __name__ == "__main__":
    # convert_data("./data/4hours.csv", "./data/4hours2.csv")
    # convert_data("./data/2hours.csv", "./data/2hours2.csv")
    train_x, train_y = data_import("/Users/zzf/Desktop/test/4hours-training.csv")
    test_x, test_y = data_import("/Users/zzf/Desktop/test/4hours-test.csv")
    nn = RecurrentNeuralNetwork()
    nn.test(train_x, train_y, test_x, test_y, batch_n=1, epochs=8)
    
