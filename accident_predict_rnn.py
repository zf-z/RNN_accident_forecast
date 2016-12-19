#!/usr/bin/python
# -*- coding: utf-8 *-*

import datetime
import os
import math
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

l_headers =["time",            # 0
            "accident num",    # 1
            "accident type",   # 2
            "holiday",         # 3
            "percipitation",   # 4
            "visibility",      # 5
            "wind",            # 6
            "wind direction",  # 7
            "fog",             # 8
            "rain",            # 9
            "sun rise",        # 10
            "sun set",         # 11
            "weekend",         # 12
            "workday",         # 13
            "t0",              # 14
            "t1",              # 15
            "t2",              # 16
            "t3"]              # 17
def data_header(cols):
    return [l_headers[x] for x in cols]

def convert_data(srcFile, dstFile, interal=2):
    if not os.path.exists(srcFile):
        return
    readCount, writeCount = 0, 0
    ac_types = {
        "NONE": "0",
        "A1": "1",
        "A2": "2",
        "A3": "3"
    }

    dfp = open(dstFile, 'w')
    with open(srcFile, 'r') as fp:
        header = fp.readline()
        # header = header.replace("Time,", "")
        # header = header.replace("time,", "")
        header = header.replace("\r\n", "")
        header = header + ",weekend,workday,t0,t1,t2,t3\r\n"
        dfp.write(header)

        for line in fp.readlines():
            readCount += 1

            values = line.split(',')
            ac_time = ""
            try:
                ac_time = datetime.datetime.strptime(values[0], "%d/%m/%Y %H:%M")
            except Exception, e1:
                try:
                    ac_time = datetime.datetime.strptime(values[0], "%H:%M %d/%m/%Y")
                except Exception, e2:
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
            line = ','.join(values)
            line = line.replace("\r\n", "")
            newLine = line + "," + strWeek + "," + strTime + "\r\n"
            dfp.write(newLine)
            writeCount += 1
        fp.close()
    dfp.close()

    if readCount != writeCount:
        print("Error:read data failed")
    print("readCount:%d, writeCount:%d" % (readCount, writeCount))


class AccStatistics:
    def __init__(self):
        self.predict_acc = 0
        self.predict_acc_tframes = 0
        self.predict_no_acc_tframes = 0

        self.total_acc = 0
        self.total_acc_tframes = 0
        self.total_no_acc_tframes = 0

    def print_scores(self):
        ScoreAccidents = float(self.predict_acc) / self.total_acc
        ScoreAccidentTFrames = float(self.predict_acc_tframes) / self.total_acc_tframes
        ScoreNonAccidents = float(self.predict_no_acc_tframes) / self.total_no_acc_tframes


        # print('predict: %d %d %d' % (
        #     self.predict_acc,
        #     self.predict_acc_tframes,
        #     self.predict_no_acc_tframes))
        # print("total  : %d %d %d" % (
        #     self.total_acc,
        #     self.total_acc_tframes,
        #     self.total_no_acc_tframes))
        description_vector = ['time','accident_num','accident_type','holiday','precipitation','visibility','wind','wind_direction','fog','rain','sun_rise','sun_set','weekend','workday','t0','t1','t2','t3']
        cols_all= range(18)
        dict_description = dict(zip(cols_all,description_vector))
        cols_choose=config["cols"]
        
        print "The Configuration vector used:"
        print "{"
        for i in range(len(cols_choose)):
            if cols_choose[i] in cols_all:
                print dict_description[cols_choose[i]]
        print "}"

        print ('Total number of accidents: {%d}' % self.total_acc)
        print ('Total number of time frames with accidents: {%d}' % self.total_acc_tframes)
        print ('Total number of non accidents: {%d}' % self.total_no_acc_tframes)

        print ('Predict:')
        print ('Total number of accidents: {%d}' % self.predict_acc)
        print ('Total number of time frames with accidents: {%d}' % self.predict_acc_tframes)
        print ('Total number of non accidents: {%d}' % self.predict_no_acc_tframes)
            
        print("ScoreAccidents:{%f}" % ScoreAccidents)
        print("ScoreNonAccidents:{%f}" % (ScoreNonAccidents))
        print("ScoreAccidents Time Frames:{%f}" % ScoreAccidentTFrames)
        print("Score1:{%f}" % ((ScoreAccidents + ScoreNonAccidents) / 2))
        print("Score2:{%f}" % (float(self.predict_acc) / (self.total_no_acc_tframes - self.predict_no_acc_tframes)))



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

    def train(self, train_x, train_y, learning_rate=0.02, epochs=1, batch_n=1, input_n=1, hidden_n=4):
        seq_n = len(train_x)
        input_n = len(train_x[0])
        output_n = len(train_y[0])

        self.inputs = tf.placeholder(tf.float32, [batch_n, input_n])
        self.label_layer = tf.placeholder(tf.float32, [output_n])
        self.input_layer = [tf.reshape(i, (1, input_n)) for i in tf.split(0, batch_n, self.inputs)]

        self.weights = tf.Variable(tf.random_normal([hidden_n, output_n]))
        self.biases = tf.Variable(tf.random_normal([output_n]))
        self.lstm_cell = rnn_cell.BasicLSTMCell(hidden_n, forget_bias=1.0)

        outputs, states = rnn.rnn(self.lstm_cell, self.input_layer, dtype=tf.float32)
        self.prediction = tf.matmul(outputs[-1], self.weights) + self.biases
        # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.prediction, self.label_layer))
        # self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        self.loss = tf.reduce_mean(tf.square(self.prediction - self.label_layer))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)

        initer = tf.global_variables_initializer()
        writer = tf.train.SummaryWriter("/Users/zzf/Desktop/test/rnn/graph-rnn", self.session.graph)

        tf.scalar_summary("loss", self.loss)
        merged_summary = tf.merge_all_summaries()

        self.session.run(initer)

        total_seq = seq_n - batch_n
        for epoch in range(epochs):
            for idx in range(0, total_seq):
                input_x = train_x[idx:idx + batch_n]
                output_y = train_y[idx]
                feed_dict = {self.inputs: input_x, self.label_layer: output_y}

                retry = 1
                #retry = 2 if int(len(train_y[idx]) >= 2) else 1

                for i in range(retry):
                    _, summary = self.session.run([self.trainer, merged_summary], feed_dict=feed_dict)

                writer.add_summary(summary, idx + epoch * total_seq)

    def test(self, test_x, test_y, batch_n, headers, times, out_file="/Users/zzf/Desktop/test/rnn/data/result.csv"):
        

        residual_nonaccident = 0.05
        residual_accident = 0.4

        seq_n = len(test_x)
        input_n = len(test_x[0])
        statistics = AccStatistics()

        acc_predict_cnt, acc_cnt = 0, 0
        no_acc_predict_cnt, no_acc_cnt = 0, 0

        with open(out_file, 'w') as fout:
            # [time/date, input vector, actual(accident), predicted(accident), result{0,1,2,3}]
            header_str = ",".join(headers)
            header_str = "time/date," + header_str
            fout.write(header_str+"\r\n")

            for idx in range(batch_n, seq_n - batch_n):
                input_x = test_x[idx:idx + batch_n]
                label_y = test_y[idx]
                predict_y = self.session.run(self.prediction, feed_dict={self.inputs: input_x})
                predict_y = predict_y[0]

                '''
                Please use this format for the result file (as discussed)
                [time/date, input vector, actual(accident), predicted(accident), result{0,1,2,3}]

                result legend-
                0, not an accident - predicted
                1, not an accident - not predicted
                2, accident - predicted
                3, accident - not predicted
                '''
                line = str(times[idx][0]) + ","
                legend = "1"
                for items in input_x:
                    items_str = [str(item) for item in items]
                    input_x_str = ",".join(items_str)
                    line += input_x_str
                    line += ","
                line += str(label_y[0]) + ","
                line += str(predict_y[0]) + ","

                if int(label_y) == 0:
                    statistics.total_no_acc_tframes += 1

                    if abs(label_y - predict_y) < residual_nonaccident:
                        no_acc_predict_cnt += 1
                        statistics.predict_no_acc_tframes += 1
                        legend = "0"
                    else:
                        legend = "1"
                else:
                    statistics.total_acc += int(label_y)
                    statistics.total_acc_tframes += 1

                    if abs(label_y - predict_y) < residual_accident:
                        statistics.predict_acc_tframes += 1
                        statistics.predict_acc += int(label_y)
                        legend = "2"
                    else:
                        legend = "3"

                line += str(legend) + "\r\n"
                fout.write(line)

            statistics.print_scores()
            fout.close()

def normalize(x):
    return (x - min(x)) / (max(x) - min(x))

def data_import(file, delimiter=',', cols=(), normalize_cols=()):
    x_cols = cols
    y_cols = (1)
    t_cols = (0)

    x = np.genfromtxt(file, delimiter=delimiter, skip_header=True, usecols=x_cols)

    for idx in range(len(cols)):
        if cols[idx] in normalize_cols:
            x[:, idx] = normalize(x[:, idx])

    y = np.genfromtxt(file, delimiter=delimiter, skip_header=True, usecols=y_cols)

    t = np.genfromtxt(file, delimiter=delimiter, skip_header=True, usecols=t_cols, dtype =[('time','<S32')])

    y = np.array([[value] for value in y])
    t = np.array([value for value in t])
    return x, y, t


if __name__ == "__main__":
    print("--> prepare data")
    if not os.path.exists("/Users/zzf/Desktop/test/rnn/data/4hours2.csv"):
        convert_data("/Users/zzf/Desktop/test/rnn/data/4hours.csv", "/Users/zzf/Desktop/test/rnn/data/4hours2.csv")
    if not os.path.exists("/Users/zzf/Desktop/test/rnn/data/2hours2.csv"):
        convert_data("/Users/zzf/Desktop/test/rnn/data/2hours.csv", "/Users/zzf/Desktop/test/rnn/data/2hours2.csv")
    dataset = {
        "train_4hours": "./data/4hours-training.csv",
        "test_4hours": "./data/4hours-test.csv",
        "train_2hours": "./data/2hours-training.csv",
        "test_2hours": "./data/2hours-test.csv",
        "4hours": "/Users/zzf/Desktop/test/rnn/data/4hours2.csv",
        "2hours": "/Users/zzf/Desktop/test/rnn/data/2hours2.csv",
    }
    config = {
        "batch_n": 1,
        "epochs": 4,
        "train_start": 0,
        "train_end": 4000,
        "cols": (2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17),
        "normalize_cols": (5, 6, 7),
    }

    headers = data_header(config["cols"])
    headers.append("actual")
    headers.append("predicted")
    headers.append("result")

    data_4hours_x, data_4hours_y, data_4hours_t = data_import(dataset["4hours"],
                                               cols=config["cols"], normalize_cols=config["normalize_cols"])
    data_2hours_x, data_2hours_y, data_2hours_t = data_import(dataset["2hours"],
                                               cols=config["cols"], normalize_cols=config["normalize_cols"])

    train_x = data_4hours_x[config["train_start"] : config["train_end"]]
    train_y = data_4hours_y[config["train_start"] : config["train_end"]]
    test_x = data_2hours_x
    test_y = data_2hours_y
    test_t = data_2hours_t

    # train_x = data_4hours_x
    # train_y = data_4hours_y
    # test_x  = data_2hours_x
    # test_y  = data_2hours_y

    print("--> trainning")
    nn = RecurrentNeuralNetwork()
    nn.train(train_x, train_y, batch_n=config["batch_n"], epochs=config["epochs"])
    print("--> testing")
    nn.test(test_x, test_y, batch_n=config["batch_n"], headers=headers, times=test_t)
    print("--> done")
