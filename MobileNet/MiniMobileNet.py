"""

@file   : MiniMobileNet.py

@author : xiaolu

@time   : 2019-07-09

"""
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical


def miniNet():
    input = Input(shape=(28, 28, 1))
    x = Conv2D(filters=32, kernel_size=(2, 2), strides=1, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D(kernel_size=(3, 3), strides=1, depth_multiplier=2, use_bias=False)(x)
    x = Conv2D(filters=128, kernel_size=(2, 2), strides=2, padding='same')(x)   # 14
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = SeparableConv2D(filters=256, kernel_size=(3, 3), strides=2, padding='same')(x)
    x = SeparableConv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')(x)
    x = SeparableConv2D(filters=1024, kernel_size=(3, 3), strides=1, padding='same')(x)

    x = GlobalAveragePooling2D()(x)
    output = Dense(10, activation='softmax')(x)

    model = Model(inputs=input, outputs=output)
    model.summary()
    plot_model(model=model, to_file='miniNet.png', show_shapes=True)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1, batch_size=64, validation_data=(x_test, y_test))


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    print(x_train.shape)  # (60000, 28, 28)
    print(x_test.shape)  # (10000, 28, 28)
    print(np.unique(y_train))  # [0 1 2 3 4 5 6 7 8 9]

    # 看一下前十张图片
    # for i in range(1, 10):
    #     plt.subplot('33{}'.format(i))
    #     plt.imshow(x_train[i])
    # plt.show()

    # 简单对数据进行处理
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_test = x_test.reshape((10000, 28, 28, 1))

    y_test = to_categorical(y_test)
    y_train = to_categorical(y_train)

    miniNet()

