"""

@file  : 栈式自编码器构建网络对mnist数据集分类.py

@author: xiaolu

@time  : 2019-08-14

"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical


# 我们使用keras中的mnist数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

tf.reset_default_graph()

x_train = x_train.reshape((-1, 28*28)).astype('float32') / 255.
x_test = x_test.reshape((-1, 28*28)).astype('float32') / 255.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

n_input = 784
n_hidden_1 = 256  # 第一层自编码
n_hidden_2 = 128  # 第二层自编码
n_classes = 10

# 训练三个自编码器 然后将它们连接起来使用

# 第一层输入
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_input])
dropout_keep_prob = tf.placeholder("float")

# 第二层输入
l2x = tf.placeholder("float", [None, n_hidden_1])
l2y = tf.placeholder("float", [None, n_hidden_1])

# 第三层输入
l3x = tf.placeholder("float", [None, n_hidden_2])
l3y = tf.placeholder("float", [None, n_classes])

# 权重初始化
weights = {
    # 网络一: 784 -> 256 -> 784
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'l1_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1])),
    'l1_out': tf.Variable(tf.random_normal([n_hidden_1, n_input])),

    # 网络二: 256 -> 128 -> 256
    'l2_h1': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'l2_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_2])),
    'l2_out': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),

    # 网络三: 128 - > 10
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'l1_b2': tf.Variable(tf.zeros([n_hidden_1])),
    'l1_out': tf.Variable(tf.zeros([n_input])),

    'l2_b1': tf.Variable(tf.zeros([n_hidden_2])),
    'l2_b2': tf.Variable(tf.zeros([n_hidden_2])),
    'l2_out': tf.Variable(tf.zeros([n_hidden_1])),

    'out': tf.Variable(tf.zeros([n_classes]))
}


# NET ONE ***************************************************************
# l1 decoder MODEL
def noise_l1_autodecoder(layer_1, _weights, _biases, _keep_prob):
    layer_1out = tf.nn.dropout(layer_1, _keep_prob)
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1out, _weights['l1_h2']), _biases['l1_b2']))
    layer_2out = tf.nn.dropout(layer_2, _keep_prob)
    return tf.nn.sigmoid(tf.matmul(layer_2out, _weights['l1_out']) + _biases['l1_out'])


# 编码输出
l1_out = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
# 解码输出
l1_reconstruction = noise_l1_autodecoder(l1_out, weights, biases, dropout_keep_prob)
# 损失
l1_cost = tf.reduce_mean(tf.pow(l1_reconstruction - y, 2))
# 优化器
l1_optimizer = tf.train.AdamOptimizer(0.01).minimize(l1_cost)


# NET TWO  ***************************************************************
def l2_autodecoder(layer1_2, _weights, _biases):
    layer1_2out = tf.nn.sigmoid(tf.add(tf.matmul(layer1_2, _weights['l2_h2']), _biases['l2_b2']))
    return tf.nn.sigmoid(tf.matmul(layer1_2out, _weights['l2_out']) + _biases['l2_out'])


# 第二层的编码输出
l2_out = tf.nn.sigmoid(tf.add(tf.matmul(l2x, weights['l2_h1']), biases['l2_b1']))
# 第二层的解码输出
l2_reconstruction = l2_autodecoder(l2_out, weights, biases)
# 损失
l2_cost = tf.reduce_mean(tf.pow(l2_reconstruction - l2y, 2))
# 优化器
l2_optimizer = tf.train.AdamOptimizer(0.01).minimize(l2_cost)


# NET THREE ***************************************************************
# l3  分类
l3_out = tf.matmul(l3x, weights['out']) + biases['out']
l3_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=l3_out, labels=l3y))
l3_optimizer = tf.train.AdamOptimizer(0.01).minimize(l3_cost)


# 将三个网络级联
# 1联2
l1_l2out = tf.nn.sigmoid(tf.add(tf.matmul(l1_out, weights['l2_h1']), biases['l2_b1']))
# 2联3
pred = tf.matmul(l1_l2out, weights['out']) + biases['out']
# 定义损失和优化器
cost3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=l3y))
optm3 = tf.train.AdamOptimizer(0.001).minimize(cost3)


epochs = 50
batch_size = 128
disp_step = 10
load_epoch = 49


batch_n = x_train.shape[0] // batch_size

'''
# 训练网络一
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("开始训练....")
    for epoch in range(epochs):
        total_cost = 0.
        for i in range(batch_n - 1):
            batch_x, batch_y = x_train[(i*batch_size):((i+1)*batch_size)], y_train[(i*batch_size): ((i+1)*batch_size)]
            batch_x_noise = batch_x + 0.3 * np.random.randn(batch_size, 784)
            feeds = {x: batch_x_noise, y: batch_x, dropout_keep_prob: 0.5}
            sess.run(l1_optimizer, feed_dict=feeds)
            total_cost += sess.run(l1_cost, feed_dict=feeds)
        if epochs % disp_step == 0:
            print('Epoch: {}/{}, average_cost:{}'.format(epoch, epochs, total_cost / batch_n))

    print(sess.run(weights['h1']))
    print(weights['h1'].name)
    print("完成")
    show_num = 10
    test_noisy = x_test[:show_num] + 0.3 * np.random.randn(show_num, 784)
    encode_decode = sess.run(l1_reconstruction, feed_dict={x: test_noisy, dropout_keep_prob: 1.})

    f, a = plt.subplots(3, 10, figsize=(10, 3))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(test_noisy[i], (28, 28)))  # 加噪声
        a[1][i].imshow(np.reshape(x_test[i], (28, 28)))   # 原始图片
        a[2][i].matshow(np.reshape(encode_decode[i], (28, 28)), cmap=plt.get_cmap('gray'))  # 去噪后的图片
    plt.show()
'''


'''
# 训练网络二
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("开始训练....")
    for epoch in range(epochs):
        total_cost = 0.
        for i in range(batch_n - 1):
            batch_x, batch_y = x_train[(i*batch_size):((i+1)*batch_size)], y_train[(i*batch_size): ((i+1)*batch_size)]

            l1_h = sess.run(l1_out, feed_dict={x: batch_x, y: batch_x, dropout_keep_prob: 1.})
            _, l2cost = sess.run([l2_optimizer, l2_cost], feed_dict={l2x: l1_h, l2y: l1_h})
            total_cost += l2cost

        if epoch % disp_step == 0:
            print('Epoch: {}/{}, average_cost:{}'.format(epoch, epochs, total_cost / batch_n))

    print(sess.run(weights['h1']))
    print(weights['h1'].name)
    print("完成")
    show_num = 10
    test_noisy = x_test[:show_num] + 0.3 * np.random.randn(show_num, 784)
    encode_decode = sess.run(l1_reconstruction, feed_dict={x: test_noisy, dropout_keep_prob: 1.})

    f, a = plt.subplots(3, 10, figsize=(10, 3))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(test_noisy[i], (28, 28)))  # 加噪声
        a[1][i].imshow(np.reshape(x_test[i], (28, 28)))   # 原始图片
        a[2][i].matshow(np.reshape(encode_decode[i], (28, 28)), cmap=plt.get_cmap('gray'))  # 去噪后的图片
    plt.show()
'''


'''
# 训练网络三
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("开始训练...")
    for epoch in range(epochs):
        total_cost = 0.
        for i in range(batch_n):
            batch_x, batch_y = x_train[(i*batch_size):((i+1)*batch_size)], y_train[(i*batch_size): ((i+1)*batch_size)]
            l1_h = sess.run(l1_out, feed_dict={x: batch_x, y: batch_x, dropout_keep_prob: 1.})               
            l2_h = sess.run(l2_out, feed_dict={l2x: l1_h, l2y: l1_h})
            _, l3cost = sess.run([l3_optimizer,l3_cost], feed_dict={l3x: l2_h, l3y: batch_y})
            total_cost += l3cost
            
        if epoch % disp_step == 0:
            print("Epoch %02d/%02d average cost: %.6f"% (epoch, epochs, total_cost/batch_n))

    print("完成layer_3训练")
    # 测试 model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(l3y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: x_test, l3y: y_test}))
'''


# 一步到位级联
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("开始训练...")
    for epoch in range(epochs):
        total_cost = 0.
        for i in range(batch_n - 1):
            batch_x, batch_y = x_train[(i*batch_size):((i+1)*batch_size)], y_train[(i*batch_size): ((i+1)*batch_size)]
            feeds = {x: batch_x, l3y: batch_y}
            sess.run(optm3, feed_dict=feeds)
            total_cost += sess.run(cost3, feed_dict=feeds)
        # 显示
        if epoch % disp_step == 0:
            print("Epoch: {}/ {}, loss: {}".format(epoch, epochs, total_cost / batch_n))
    print("完成级联操作")

    # 测试 model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(l3y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: x_test, l3y: y_test}))
