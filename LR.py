"""
Created on Tue Apr 10 21:47:30 2018
基于tensorflow框架用logstic regeression实现minist数据分类
参考http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_beginners.html网址稍加改动
@author: zcz
"""
#导入数据
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf      
x = tf.placeholder(tf.float32, [None, 784])  #x是占位符，存储mnist图像
W = tf.Variable(tf.zeros([784,10]))          #权重
b = tf.Variable(tf.zeros([10])   )            #bias
y = tf.nn.sigmoid(tf.matmul(x,W) + b)       #输出标签
#y = tf.nn.softmax(tf.matmul(x,W) + b)     #这里换成softmax回归也可以
y_ = tf.placeholder("float", [None,10])     #真实标签
cross_entropy = -tf.reduce_sum(y_*tf.log(y)+(1-y_)*tf.log(1-y)) #似然函数的log形式，常称为loglikelihood
#cross_entropy = -tf.reduce_sum(y_*tf.log(y))  #softmax回归下的交叉熵函数
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  #梯度下降优化函数
init = tf.initialize_all_variables()   #初始化

sess = tf.Session()   #启动一个tensorflow会话
sess.run(init)        #运行初始化
for i in range(1000):  #训练过程
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
 
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))  #正确与否
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) #正确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

