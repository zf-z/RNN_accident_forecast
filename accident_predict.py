zhe#!/usr/bin/python
# -*- coding: utf-8 *-*

import numpy as np
import datetime
import os

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.python.ops import rnn, rnn_cell

def convert_data(srcFile, dstFile, interal=2):
    if not os.path.exists(srcFile):
        return
    readCount,writeCount = 0,0
    ac_types ={
      "NONE":"0",
      "A1"  :"1",
      "A2"  :"2",
      "A3"  :"3" 
    }

    dfp = open(dstFile, 'w')
    with open(srcFile, 'r') as fp:
        header = fp.readline()
        header = header.replace("\r\n", "")
        header = header + ",weekend,workday,t0,t1,t2,t3\r\n"
        dfp.write(header)

        for line in fp.readlines():
            readCount += 1

            values = line.split(',')
            ac_time = ""
            try:
                ac_time = datetime.datetime.strptime(values[0], "%d/%m/%Y %H:%M")
            except Exception,e1:
                try:
                    ac_time = datetime.datetime.strptime(values[0], "%H:%M %d/%m/%Y")
                except Exception,e2:
                    continue

            strWeek = "0,0"
            strTime = "0,0,0,0"
            if ac_time.weekday() >= 5:
                strWeek = "1,0"
            else:
                strWeek = "0,1"
            if ac_time.hour <= 6 or ac_time.hour >= 22:
                strTime = "1,0,0,0"
            elif ac_time.hour >= 6 and ac_time.hour <= 10:
                strTime = "0,1,0,0"
            elif ac_time.hour >= 10 and ac_time.hour <= 16:
                strTime = "0,0,1,0"
            elif ac_time.hour >= 16 and ac_time.hour <= 22:
                strTime = "0,0,0,1"

            # convert ac_type
            values[2] = ac_types[values[2]]
            line = ','.join(values[1:])
            line = line.replace("\r\n","")
            newLine = line + "," + strWeek + "," + strTime + "\r\n"
            dfp.write(newLine)
            writeCount += 1
        fp.close()
    dfp.close()

    if(readCount != writeCount):
        print("Error:read data failed")
    print("readCount:%d, writeCount:%d"%(readCount,writeCount))

class RecurrentNeuralNetwork:
    def __init__(self):
        self.session = tf.Session()
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

    def train(self, train_x, train_y, learning_rate=0.05, limit=1000, batch_n=4, input_n=2, hidden_n=5):
        seq_len = len(train_x)
        input_n = len(train_x[0])
        output_n = 1

        in_shape  = [seq_len, input_n]
        out_shape = [seq_len, output_n] 

        #self.input_layer = tf.placeholder(tf.float32, in_shape)
        self.input_layer = [tf.placeholder(tf.float32, in_shape) for i in range(batch_n)]
        self.label_layer = tf.placeholder(tf.float32, out_shape)

        # [hidden_n ,output_n]?
        self.weights = tf.Variable(tf.random_normal([hidden_n, output_n]))
        self.biases = tf.Variable(tf.random_normal([output_n]))
        self.lstm_cell = rnn_cell.BasicLSTMCell(hidden_n, forget_bias=1.0)

        outputs, states = rnn.rnn(self.lstm_cell, self.input_layer, dtype=tf.float32)
        # new_output = output*w + bias
        self.prediction = tf.matmul(outputs[-1], self.weights) + self.biases
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.prediction, self.label_layer))
        self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        initer = tf.global_variables_initializer()
        
        train_x = train_x.reshape((batch_n, seq_len, input_n))
        writer = tf.train.SummaryWriter("./graph", self.session.graph)
        print(train_x)
        
        tf.scalar_summary("loss", self.loss)
        merged_summary = tf.merge_all_summaries()

        # run graph
        self.session.run(initer)
        for i in range(limit):
            _,summary = self.session.run([self.trainer,merged_summary], feed_dict={self.input_layer[0]: train_x[0], self.label_layer: train_y})
            writer.add_summary(summary, i)

    def predict(self, test_x):
        return self.session.run(self.prediction, feed_dict={self.input_layer[0]: test_x})

    def test(self, train_x, train_y, test_x, test_y):
        self.train(train_x, train_y, batch_n=1)
        out = self.predict(test_x)
        print(len(out))
        print(out)

'''
  data_import
    dtype =[
            #('time'         ,'<S32'),   # 0  time
            ('ac_num'       ,int),      # 1  ac_num
            ('ac_type'      ,int),      # 2  ac_type
            ('holiday'      ,int),      # 3  holiday
            ('prec'         ,float),    # 4  prec
            ('visibility'   ,int),      # 5  visibility
            ('wind'         ,float),    # 6  wind
            ('wind_dir'     ,int),      # 7  wind_dir
            ('fog'          ,int),     # 8  fog
            ('rain'         ,int),     # 9  rain
            ('sun_rise'     ,int),     # 10 sun_rise
            ('sun_set'      ,int),     # 11 sun_set
            ('weekend'      ,int),     # 12 weekend
            ('workday'      ,int),     # 13 workday
            ('t0'           ,int),     # 14 t0
            ('t1'           ,int),     # 15 t1
            ('t2'           ,int),     # 16 t2
            ('t3'           ,int),     # 17 t3
    ]
'''
def data_import(file, delimiter=','):
    x_cols = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)
    y_cols = (0)
    x = np.genfromtxt(file, delimiter=delimiter,skip_header=True,usecols=x_cols)
    y = np.genfromtxt(file, delimiter=delimiter,skip_header=True,usecols=y_cols)
    y = np.array([[value] for value in y])
    return x,y

if __name__ == "__main__":
    #convert_data("./data/4hours.csv", "./data/4hours2.csv")
    #convert_data("./data/2hours.csv", "./data/2hours2.csv")
    
    train_x,train_y = train_data = data_import("/Users/zzf/Desktop/test/4hours-training.csv")
    test_x,test_y = train_data = data_import("/Users/zzf/Desktop/test/4hours-test.csv")
    print("training shape: x%s, y%s"%(str(train_x.shape),str(train_y.shape)))
    print("test shape: x%s, y%s"%(str(test_x.shape),str(test_y.shape)))
    print("type(train_x)=%s"%(type(train_x)))

    nn = RecurrentNeuralNetwork()
    nn.test(train_x, train_y,test_x, test_y)
    #shape = [len(train_x), len(train_x[0])]
    #print(shape)

    #print(len(test_x))
    #print(len(test_y))

