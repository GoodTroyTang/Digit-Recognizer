# -*- coding: utf-8 -*-
"""
pelven
3层神经网络，隐藏层节点数为500
"""

import csv
import numpy as np
#import pandas as pd
import tensorflow as tf

INPUT_NODE = 784  #输入节点数
OUTPUT_NODE = 10  #输出节点数

##配置神经网络参数
LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8  #基础学习率
LEARNING_RATE_DECAY = 0.99  #学习率的衰减率
TRAINING_STEPS = 3000  #训练轮数
REGULARIZATION_RATE = 0.0001 #正则化项系数


##读取训练数据，并划分为训练集和验证集
def loadTrainData():
    train_images = []
    train_labels = []
    validation_images = []
    validation_labels = []
    validation_percentage = 20  #将20%的训练数据划分为验证集
    
    csvFile = open("train.csv")
    lines = csv.reader(csvFile)
    next(lines)  #跳过第一行（属性行）
    
    for line in lines:
        ##data = np.array(line)
        lineArr = []
        for i in range(1,785):
            lineArr.append(np.float32(line[i])/255)
        #img_data = tf.image.convert_image_dtype(lineArr[1:],dtype=tf.float32)
        chance = np.random.randint(100)
        if chance >= validation_percentage:
            train_images.append(lineArr)
            train_labels.append(int(line[0]))
        else:
            validation_images.append(lineArr)
            validation_labels.append(int(line[0]))
            
    #将训练数据随机打乱以获得更好的训练效果
    state = np.random.get_state()
    np.random.shuffle(train_images)
    np.random.set_state(state)
    np.random.shuffle(train_labels)
    
    return train_images,train_labels,validation_images,validation_labels



##计算神经网络的前向传播过程
def inference(input_tensor, weights1, biases1, weights2, biases2):
    layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1) + biases1)
    layerOut = tf.matmul(layer1,weights2) + biases2
    return layerOut

##训练模型的过程
def train(train_images, train_labels, validation_images, validation_labels):
    x = tf.placeholder(tf.float32,[None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32,[None, OUTPUT_NODE], name='y-input')
    
    #生成隐藏层参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    #生成输出层参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
    
    #计算当前神经网络下的前向传播过程
    y = inference(x, weights1,biases1,weights2,biases2)
    
    #训练轮数
    global_step = tf.Variable(0, trainable=False)
    
    #得到可变参数
    variables = tf.trainable_variables()
    
    #计算交叉熵，衡量预测值与真实值之间差距的损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    #计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1) + regularizer(weights2)
    #总损失等于交叉熵损失与正则化损失的和
    loss = cross_entropy_mean + regularization
    
    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            400,
            LEARNING_RATE_DECAY)
    
    #使用tf.train.GradientDescentOptimizer优化算法来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    
    #更新参数
    train_op = tf.group(train_step, variables)
    
    #检验神经网络前向传播结果是否正确
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    start = 0
    end = BATCH_SIZE
    train_number = np.shape(train_labels)[0]
    #初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        
        #验证集
        y_validation = tf.one_hot(validation_labels, 10, on_value=1,off_value=None)
        y_validation = sess.run(y_validation)
        validate_feed = {x: validation_images,
                         y_: y_validation}
    
        for i in range(TRAINING_STEPS):
            if (i+1)%10 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("train steps: %d, validation accuracy: %g"%(i+1,validate_acc))
            
            #产生这一轮的训练数据
            xs  = train_images[start:end]
            ys = train_labels[start:end]
            ys = tf.one_hot(ys, 10, on_value=1,off_value=None)
            ys = sess.run(ys)
            sess.run(train_op, feed_dict={x:xs, y_:ys})
        
            start = end
            if start == train_number:
                start = 0
            
            end = start + BATCH_SIZE
            if end > train_number:
                end = train_number
                
        w1 = sess.run(weights1)
        b1 = sess.run(biases1)
        w2 = sess.run(weights2)
        b2 = sess.run(biases2)
        sess.close()
    #返回模型参数
    return w1, b1, w2, b2 

##利用得到的神经网络参数对测试集进行预测   
def test(weights1, biases1, weights2, biases2):
    testFile = open("test.csv")
    lines = csv.reader(testFile)
    next(lines)  #跳过第一行（属性行）
    
    test_number = 0
    
    #testResult = open("test_submission.csv","w") 
    #writer = csv.writer(testResult)
    test_images = []
    #testResult = []
    for line in lines:
        ##data = np.array(line)
        test_number += 1
        lineArr = []
        for i in range(784):
            lineArr.append(np.float32(line[i])/255)
        test_images.append(lineArr)
    
    with tf.Session() as sess:
        y_ = inference(test_images, weights1, biases1, weights2, biases2)
        y = tf.argmax(y_, 1)
        y = sess.run(y)
        sess.close()
    return y

##将预测结果写入csv文件
def writeCsv(y):
    test_number = np.shape(y)[0]
    test_result = []
    for i in range(test_number):
        test_result.append([i+1,y[i]])
        
    with open("test_submission.csv","w",newline='') as subFile:
        writer = csv.writer(subFile)
        writer.writerow(["ImageId","Label"])
        writer.writerows(test_result)
   
            

          
##主程序入口
def main(argv=None):
    train_images, train_labels, validation_images, validation_labels = loadTrainData()
    w1, b1, w2, b2 = train(train_images, train_labels, validation_images, validation_labels)
    #return w1, b1,w2, b2
    y = test(w1,b1,w2,b2)
    writeCsv(y)
    
if __name__ == '__main__':
    tf.app.run()
    
            
        
            