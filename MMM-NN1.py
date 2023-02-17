# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 16:19:28 2022

@author: Admin
"""

import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io as scio
import os
import datetime

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

tf.disable_eager_execution()

# data for save
AllData = scio.loadmat('C:/Users/Admin/Desktop/Train/Train-1.mat')

TrainData = AllData['TrainData']  # [sample, inputs]
TestData = AllData['TestData']
TrainResult = AllData['TrainLabel']  # [sample, outputs]
TestResult = AllData['TestLabel']

NumTaste = 6
NumTrain = TrainData.shape[0]
NumInput = TrainData.shape[1]
NumTest = TestData.shape[0]

NumNeurons = 512
KeepProb = 1
lr = 1.5 * 1e-6
Iteration = 40000

# MultiModel = np.array([[0.0], [5], [10]], 'float32')
MultiModel = np.array([[-5], [0], [5]], 'float32')
NumModel = MultiModel.shape[0]

TrainLoss = np.zeros([NumTaste, NumTrain, Iteration])
TestLoss = np.zeros([NumTaste, NumTest, Iteration])

TrainAllLoss = np.zeros([NumTaste, Iteration])
TestAllLoss = np.zeros([NumTaste, Iteration])
TrainSquareLoss = np.zeros([NumTaste, NumTrain, Iteration])
TestSquareLoss = np.zeros([NumTaste, NumTest, Iteration])

TrainPred = np.zeros([NumTaste, NumTrain, Iteration])
TestPred = np.zeros([NumTaste, NumTest, Iteration])

StartTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

# for Taste in [5]:
for Taste in [0, 1, 2, 3, 4, 5]:
    tf.reset_default_graph()  # reset the graph

    FileName = 'C:/Users/Admin/Desktop/Train/Train-' + str(Taste+1) + '.mat'
    # File = FolderPath + r'/CuiCui/Data/SISO_1_1_20210714T195500.mat'
    # File = FolderPath + r'/CuiCui/Data/' + FileName
    AllData = scio.loadmat(FileName)

    TrainData = AllData['TrainData']  # [sample, inputs]
    TestData = AllData['TestData']
    TrainResult = AllData['TrainLabel']  # [sample, outputs]
    TestResult = AllData['TestLabel']

    x = tf.placeholder('float32', [None, NumInput], name='x')
    y = tf.placeholder('float32', [None, 1], name='y')

    mean = 0.001
    stddev = 0.002

    WeightIn = tf.Variable(tf.random_normal([NumInput, NumNeurons], mean=mean, stddev=stddev))  # mean, stddev, 2
    BiaIn = tf.Variable(tf.random_normal([1, NumNeurons], mean=mean, stddev=stddev))  # mean, stddev

    WeightHidden = tf.Variable(tf.random_normal([NumNeurons, NumNeurons], mean=mean, stddev=stddev))  # mean, stddev
    BiaHidden = tf.Variable(tf.random_normal([1, NumNeurons], mean=mean, stddev=stddev))  # mean, stddev

    WeightOut = tf.Variable(tf.random_normal([NumNeurons, NumModel], mean=mean, stddev=stddev))  # mean, stddev
    BiaOut = tf.Variable(tf.random_normal([1, NumModel], mean=mean, stddev=stddev))  # mean, stddev

    hiddena = tf.nn.tanh(tf.matmul(x, WeightIn) + BiaIn)
    hiddena = tf.nn.dropout(hiddena, keep_prob=KeepProb)
    hiddenb = tf.nn.tanh(tf.matmul(hiddena, WeightHidden) + BiaHidden)
    hiddenb = tf.nn.dropout(hiddenb, keep_prob=KeepProb)
    hiddenc = tf.nn.tanh(tf.matmul(hiddenb, WeightHidden) + BiaHidden)
    hiddenc = tf.nn.dropout(hiddenc, keep_prob=KeepProb)
    output = tf.nn.softmax(tf.matmul(hiddenb, WeightOut) + BiaOut)
    output = tf.nn.dropout(output, keep_prob=KeepProb)
    pred = tf.matmul(output, MultiModel)
    print(pred)
    Pred = tf.reshape(pred, [-1])
    # print(Pred)
    print(y)

    AbsLoss = tf.reshape(pred - y, [-1])
    print(AbsLoss)
    SquareLoss = tf.square(AbsLoss)
    print(SquareLoss)
    Loss = tf.reduce_mean(SquareLoss)
    # print(AbsLoss)

    TrainStep = tf.train.AdamOptimizer(lr).minimize(Loss)

    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    print('Start Training!')

    for i in range(Iteration):
    # for i in range(Iteration):
        sess.run(TrainStep, {x: TrainData, y: TrainResult})

        TrainLoss[Taste, :, i] = sess.run(AbsLoss, {x: TrainData, y: TrainResult})
        TestLoss[Taste, :, i] = sess.run(AbsLoss, {x: TestData, y: TestResult})

        TrainSquareLoss[Taste, :, i] = sess.run(SquareLoss, {x: TrainData, y: TrainResult})
        TestSquareLoss[Taste, :, i] = sess.run(SquareLoss, {x: TestData, y: TestResult})

        TrainAllLoss[Taste, i] = sess.run(Loss, {x: TrainData, y: TrainResult})
        TestAllLoss[Taste, i] = sess.run(Loss, {x: TestData, y: TestResult})

        TrainPred[Taste, :, i] = sess.run(Pred, {x: TrainData, y: TrainResult})
        TestPred[Taste, :, i] = sess.run(Pred, {x: TestData, y: TestResult})

        if (i+1) % 500 == 0:
            # print('Iteration:', i, ',TrainLoss:', TrainAllLoss[Taste, i], ',TestLoss:', TestAllLoss[Taste, i])
            print('Taste:', Taste+1,
                  'Iteration:', i+1,
                  ';TrainLoss:', TrainAllLoss[Taste, i],
                  # ';TestLoss:', TestAllLoss[Taste, i],
                  # ';TestSquareLoss:', TestSquareLoss[Taste, :, i],
                  ';TestLoss:', TestLoss[Taste, :, i])
            if TestAllLoss[Taste, i] < 0.04:
                saver.save(sess, r'./Model/Model-' + str(Taste) + '-' + str(NumNeurons) + '-' + str(Iteration) + str(StartTime))
# print(TestLoss[:, :, -1])
# print('Taste:', Taste+1,
#       ';TrainLoss:', TrainAllLoss[Taste, i],
#       ';TestLoss:', TestAllLoss[Taste, i],
#       TestLoss[Taste, :, i])

# MeanLoss = np.sum(MSELoss[:, :, -1]) / 35
scio.savemat('MISO-' + str(StartTime) + '.mat',
             {'TrainLoss': TrainLoss,
              'TestLoss': TestLoss,
              'TrainAllLoss': TrainAllLoss,
              'TestAllLoss': TestAllLoss,
              'TrainPred': TrainPred,
              'TestPred': TestPred})
# print('Mean:', MeanLoss)

# saver.save(sess, r'./Model/SISO' + str(StartTime))







