"""

@file  : DC_GAN&INFO_GAN&AC_GAN.py

@author: xiaolu

@time  : 2019-08-14

"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from keras.datasets import mnist


# 加载数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((-1, 28*28)).astype('float32') / 255.
x_test = x_test.reshape((-1, 28*28)).astype('float32') / 255.

tf.reset_default_graph()


def generator(x):
    '''
    定义生成式模型
    :param x:
    :return:
    '''
    reuse = len([t for t in tf.global_variables() if t.name.startswith('generator')]) > 0
    with tf.variable_scope('generator', reuse=reuse):
        x = slim.fully_connected(x, 1024)
        x = slim.batch_norm(x, activation_fn=tf.nn.relu)

        x = slim.fully_connected(x, 7 * 7 * 128)
        x = slim.batch_norm(x, activation_fn=tf.nn.relu)

        x = tf.reshape(x, [-1, 7, 7, 128])

        x = slim.conv2d_transpose(x, 64, kernel_size=[4, 4], stride=2, activation_fn=None)
        print(x.get_shape())

        x = slim.batch_norm(x, activation_fn=tf.nn.relu)
        z = slim.conv2d_transpose(x, 1, kernel_size=[4, 4], stride=2, activation_fn=tf.nn.sigmoid)
        print(z.get_shape())
    return z


def leaky_relu(x):
    # 定义受限的relu函数
    return tf.where(tf.greater(x, 0), x, 0.01 * x)


def discriminator(x, num_classes=10, num_cont=2):
    '''
    定义判别式模型
    :param x:
    :param num_classes: 类别
    :param num_cont: 附加的额外信息
    :return: disc: 真假预测, recog_cat: 类别预测, recog_cont: 额外信息的预测
    '''
    reuse = len([t for t in tf.global_variables() if t.name.startswith('discriminator')]) > 0
    with tf.variable_scope('discriminator', reuse=reuse):
        x = tf.reshape(x, shape=(-1, 28, 28, 1))
        x = slim.conv2d(x, num_outputs=64, kernel_size=[4, 4], stride=2, activation_fn=leaky_relu)
        x = slim.conv2d(x, num_outputs=128, kernel_size=[4, 4], stride=2, activation_fn=leaky_relu)

        x = slim.flatten(x)
        shared_tensor = slim.fully_connected(x, num_outputs=1024, activation_fn=leaky_relu)
        recog_shared = slim.fully_connected(shared_tensor, num_outputs=128, activation_fn=leaky_relu)

        # 判别真假
        disc = slim.fully_connected(shared_tensor, num_outputs=1, activation_fn=None)

        # 分类
        disc = tf.squeeze(disc, -1)
        recog_cat = slim.fully_connected(recog_shared, num_outputs=num_classes, activation_fn=None)

        # 拟合附加的额外信息
        recog_cont = slim.fully_connected(recog_shared, num_outputs=num_cont, activation_fn=tf.nn.sigmoid)

    return disc, recog_cat, recog_cont


batch_size = 10  # 获取样本的批次大小32
classes_dim = 10  # 10 classes
con_dim = 2  # total continuous factor
rand_dim = 38
n_input = 784


x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.int32, [None])

z_con = tf.random_normal((batch_size, con_dim))  # 构造附加信息
z_rand = tf.random_normal((batch_size, rand_dim))  # 2列
z = tf.concat(axis=1, values=[tf.one_hot(y, depth=classes_dim), z_con, z_rand])  # 50列
gen = generator(z)
genout = tf.squeeze(gen, -1)


# labels for discriminator
y_real = tf.ones(batch_size)  # 真
y_fake = tf.zeros(batch_size)  # 假


# 判别器
disc_real, class_real, _ = discriminator(x)   # 真数据
disc_fake, class_fake, con_fake = discriminator(gen)  # 假数据
pred_class = tf.argmax(class_fake, dimension=1)


# 判别器 loss
loss_d_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=y_real))
loss_d_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=y_fake))
loss_d = (loss_d_r + loss_d_f) / 2
# print ('loss_d', loss_d.get_shape())


# 生成器 loss
loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=y_real))
loss_cf = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=class_fake, labels=y))
loss_cr = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=class_real, labels=y))
loss_c = (loss_cf + loss_cr) / 2


# 额外信息的损失 直接用均方误差损失
loss_con = tf.reduce_mean(tf.square(con_fake - z_con))

# 获得各个网络中各自的训练参数
t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'discriminator' in var.name]
g_vars = [var for var in t_vars if 'generator' in var.name]


disc_global_step = tf.Variable(0, trainable=False)
gen_global_step = tf.Variable(0, trainable=False)

train_disc = tf.train.AdamOptimizer(0.0001).minimize(loss_d + loss_c + loss_con, var_list=d_vars,
                                                     global_step=disc_global_step)
train_gen = tf.train.AdamOptimizer(0.001).minimize(loss_g + loss_c + loss_con, var_list=g_vars,
                                                   global_step=gen_global_step)

training_epochs = 3
display_step = 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = x_train.shape[0] // batch_size

        # 遍历全部数据集
        for i in range(total_batch-1):
            batch_xs, batch_ys = x_train[(i*batch_size):((i+1)*batch_size)], y_train[(i*batch_size):((i+1)*batch_size)]  # 取数据
            feeds = {x: batch_xs, y: batch_ys}

            # Fit training using batch data
            l_disc, _, l_d_step = sess.run([loss_d, train_disc, disc_global_step], feeds)
            l_gen, _, l_g_step = sess.run([loss_g, train_gen, gen_global_step], feeds)

        # 显示训练中的详细信息
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f} ".format(l_disc), l_gen)

    print("完成!")

    # 测试
    print("Result:", loss_d.eval({x: x_test[:batch_size], y: y_test[:batch_size]})
          , loss_g.eval({x: x_test[:batch_size], y: y_test[:batch_size]}))

    # 根据图片模拟生成图片
    show_num = 10
    gensimple, d_class, inputx, inputy, con_out = sess.run(
        [genout, pred_class, x, y, con_fake],
        feed_dict={x: x_test[:batch_size], y: y_test[:batch_size]})

    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(inputx[i], (28, 28)))
        a[1][i].imshow(np.reshape(gensimple[i], (28, 28)))
        print("d_class", d_class[i], "inputy", inputy[i], "con_out", con_out[i])

    plt.draw()
    plt.show()

    my_con = tf.placeholder(tf.float32, [batch_size, 2])
    myz = tf.concat(axis=1, values=[tf.one_hot(y, depth=classes_dim), my_con, z_rand])
    mygen = generator(myz)
    mygenout = tf.squeeze(mygen, -1)

    my_con1 = np.ones([10, 2])
    a = np.linspace(0.0001, 0.99999, 10)
    y_input = np.ones([10])
    figure = np.zeros((28 * 10, 28 * 10))
    my_rand = tf.random_normal((10, rand_dim))
    for i in range(10):
        for j in range(10):
            my_con1[j][0] = a[i]
            my_con1[j][1] = a[j]
            y_input[j] = j
        mygenoutv = sess.run(mygenout, feed_dict={y: y_input, my_con: my_con1})
        for jj in range(10):
            digit = mygenoutv[jj].reshape(28, 28)
            figure[i * 28: (i + 1) * 28,
            jj * 28: (jj + 1) * 28] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()