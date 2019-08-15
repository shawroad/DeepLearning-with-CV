"""

@file  : 条件变分自编码器.py

@author: xiaolu

@time  : 2019-08-15

"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.datasets import mnist
from keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_test_ = y_test
x_train = x_train.reshape((-1, 28*28)).astype('float32') / 255.
x_test = x_test.reshape((-1, 28*28)).astype('float32') / 255.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


n_labels = 10
n_input = 784
n_hidden_1 = 256
n_hidden_2 = 2

input_x = tf.placeholder(tf.float32, [None, n_input])
input_y = tf.placeholder(tf.float32, [None, n_labels])

zinput = tf.placeholder(tf.float32, [None, n_hidden_2])

weights = {
    # 1. 数据线性变换为一定长度的向量
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], stddev=0.001)),
    'b1': tf.Variable(tf.zeros([n_hidden_1])),

    # 2. 标签变换为一定长度的向量
    'wlab1': tf.Variable(tf.truncated_normal([n_labels, n_hidden_1], stddev=0.001)),
    'blab1': tf.Variable(tf.zeros([n_hidden_1])),

    # 3. 数据转为的向量和标签变为的向量进行拼接  得到均值和方差
    'mean_w1': tf.Variable(tf.truncated_normal([n_hidden_1 * 2, n_hidden_2], stddev=0.001)),
    'log_sigma_w1': tf.Variable(tf.truncated_normal([n_hidden_1 * 2, n_hidden_2], stddev=0.001)),

    # 4. 解码　　将均值和方差中填入标签
    'w2': tf.Variable(tf.truncated_normal([n_hidden_2 + n_labels, n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_1])),

    # 5. 在通过一层线性映射得到最后的图片大小
    'w3': tf.Variable(tf.truncated_normal([n_hidden_1, n_input], stddev=0.001)),
    'b3': tf.Variable(tf.zeros([n_input])),

    'mean_b1': tf.Variable(tf.zeros([n_hidden_2])),
    'log_sigma_b1': tf.Variable(tf.zeros([n_hidden_2]))

}

# 数据和标签转为同样长度的向量
h1 = tf.nn.relu(tf.add(tf.matmul(input_x, weights['w1']), weights['b1']))
hlab1 = tf.nn.relu(tf.add(tf.matmul(input_y, weights['wlab1']), weights['blab1']))

# 将两个向量进行拼接
hall1 = tf.concat([h1, hlab1], 1)  # 256*2

# 再来一个线性变换 得到均值和方差的对数
z_mean = tf.add(tf.matmul(hall1, weights['mean_w1']), weights['mean_b1'])
z_log_sigma_sq = tf.add(tf.matmul(hall1, weights['log_sigma_w1']), weights['log_sigma_b1'])


# 重参数技巧 进行采样
eps = tf.random_normal(tf.stack([tf.shape(h1)[0], n_hidden_2]), 0, 1, dtype=tf.float32)
z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))

# 将标签和采样的结果进行拼接
zall = tf.concat([z, input_y], 1)

# 经过一个线性变化, 然后在经过一个线性变化重构图像
h2 = tf.nn.relu(tf.matmul(zall, weights['w2']) + weights['b2'])
reconstruction = tf.matmul(h2, weights['w3']) + weights['b3']

# 当网络训练好 我们是从这个位置开始做起
zinputall = tf.concat([zinput, input_y], 1)
h2out = tf.nn.relu(tf.matmul(zinputall, weights['w2']) + weights['b2'])
reconstructionout = tf.matmul(h2out, weights['w3']) + weights['b3']


# 定义损失函数
# 1. 重构误差
reconstruction_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(reconstruction, input_x), 2.0))
# 2. KL损失
latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
cost = tf.reduce_mean(reconstruction_loss + latent_loss)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

training_epochs = 50
batch_size = 128
display_step = 3


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("开始训练．．．")
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = x_train.shape[0] // batch_size

        for i in range(total_batch - 1):
            batch_x, batch_y = x_train[(i*batch_size): ((i + 1)*batch_size)], y_train[(i*batch_size): ((i + 1)*batch_size)]
            _, c = sess.run([optimizer, cost], feed_dict={input_x: batch_x, input_y: batch_y})

            # _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
            # c = autoencoder.partial_fit(batch_xs)

        # 显示训练中的详细信息
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

    print("完成!")

    # 测试
    print('Result:', cost.eval({input_x: x_test, input_y: y_test}))

    # 根据图片模拟生成图片　挑选测试集中前10个图片　然后画出原始的图和重构之后的图
    show_num = 10
    pred = sess.run(
        reconstruction,
        feed_dict={input_x: x_test[:show_num], input_y: y_test[:show_num]})

    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(x_test[i], (28, 28)))
        a[1][i].imshow(np.reshape(pred[i], (28, 28)))
    plt.draw()

    # 将label和随机生成的均值和方差拼接到一块 直接获取对应的图片
    # 根据label模拟生产图片可视化结果
    show_num = 10
    z_sample = np.random.randn(10, 2)   # 生成均值和方差　

    pred = sess.run(
        reconstructionout, feed_dict={zinput: z_sample, input_y: y_test[:show_num]})

    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(x_test[i], (28, 28)))  # 画出测试集的前是个图
        a[1][i].imshow(np.reshape(pred[i], (28, 28)))  # 根测试集前十个lable生成对应的图片
    plt.draw()
    plt.show()

    '''
    pred = sess.run(z, feed_dict={input_x: x_test})
    plt.figure(figsize=(6, 6))
    plt.scatter(pred[:, 0], pred[:, 1], c=y_test_)
    plt.colorbar()
    plt.draw()

    # 再进行画图
    n = 15
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = sess.run(reconstructionout, feed_dict={zinput: z_sample})
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j+1)*digit_size] = digit
    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()
    '''
