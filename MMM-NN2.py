# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 17:09:03 2022

@author: Admin
"""

import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io as scio
import os
import datetime

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.disable_eager_execution()

# data for save
AllData = scio.loadmat('C:/Users/Admin/Desktop/TestData/Test-20-100-5-10-1-20221018T195531.mat')
TestData = AllData['TestData']  # [sample, inputs]

NumTaste = 6
NumTest = TestData.shape[0]
NumInput = TestData.shape[1]
Score = np.zeros([NumTest, NumTaste])
# Score = np.zeros([NumTest*9, NumTaste])

NumNeurons = 512
KeepProb = 1
lr = 1.5 * 1e-3

# MultiModel = np.array([[0.0], [5], [10]], 'float32')
MultiModel = np.array([[-6], [0], [6]], 'float32')
NumModel = MultiModel.shape[0]

StartTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

for i in range(9):
    tf.reset_default_graph()  # reset the graph

    x = tf.placeholder('float32', [None, NumInput], name='x')
    y = tf.placeholder('float32', [None, 1], name='y')

    WeightIn = tf.Variable(tf.random_normal([NumInput, NumNeurons], mean=0.001, stddev=0.002))  # mean, stddev, 2
    BiaIn = tf.Variable(tf.random_normal([1, NumNeurons], mean=0.001, stddev=0.002))  # mean, stddev

    WeightHidden = tf.Variable(tf.random_normal([NumNeurons, NumNeurons], mean=0.001, stddev=0.002))  # mean, stddev
    BiaHidden = tf.Variable(tf.random_normal([1, NumNeurons], mean=0.001, stddev=0.002))  # mean, stddev

    WeightOut = tf.Variable(tf.random_normal([NumNeurons, NumModel], mean=0.001, stddev=0.001))  # mean, stddev
    BiaOut = tf.Variable(tf.random_normal([1, NumModel], mean=0.001, stddev=0.001))  # mean, stddev

    hiddena = tf.nn.tanh(tf.matmul(x, WeightIn) + BiaIn)
    hiddena = tf.nn.dropout(hiddena, keep_prob=KeepProb)
    hiddenb = tf.nn.tanh(tf.matmul(hiddena, WeightHidden) + BiaHidden)
    hiddenb = tf.nn.dropout(hiddenb, keep_prob=KeepProb)
    hiddenc = tf.nn.tanh(tf.matmul(hiddenb, WeightHidden) + BiaHidden)
    hiddenc = tf.nn.dropout(hiddenc, keep_prob=KeepProb)
    output = tf.nn.softmax(tf.matmul(hiddenb, WeightOut) + BiaOut)
    output = tf.nn.dropout(output, keep_prob=KeepProb)
    pred = tf.matmul(output, MultiModel)
    Pred = tf.reshape(pred, [-1])

    AbsLoss = tf.reshape(pred - y, [1, -1])
    Loss = tf.reduce_mean(tf.square(pred - y))
    TrainStep = tf.train.AdamOptimizer(lr).minimize(Loss)

    # for i in range(NumTaste):
    for Taste in range(NumTaste):
        FileName = 'C:/Users/Admin/Desktop/TestData/Test-20-100-5-10-' + str(i+1) + '-20221018T195531.mat'
        AllData = scio.loadmat(FileName)
        TestData = AllData['TestData']  # [sample, inputs]
        saver = tf.train.Saver()
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'C:/Users/Admin/Desktop/Model/Model-' + str(Taste) + '-512-400002022-02-23-20-54-49')
        print('Start Training!')
        Score[:, Taste] = sess.run(Pred, {x: TestData})
        # Score[range(i*NumTest, (i+1)*NumTest), Taste] = sess.run(Pred, {x: TestData})

        # MeanLoss = np.sum(MSELoss[:, :, -1]) / 35
    scio.savemat('C:/Users/Admin/Desktop/MISOresult/MISO-Test-' + str(i) + '-' + str(StartTime) + '.mat',
                 {'Score': Score})
# print('Mean:', MeanLoss)

# saver.save(sess, r'./Model/SISO' + str(StartTime))
