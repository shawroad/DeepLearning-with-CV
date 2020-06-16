"""

@file  : costvolume.py

@author: xiaolu

@time  : 2020-01-17

"""
import torch
import numpy as np


def CostVolume(input_feature, candidate_feature, position='left', method='subtract',
               k=4, batch_size=4, channel=32, D=192, H=256, W=512):
    '''
    :param input_feature:
    :param candidate_feature:
    :param positiom: means whether the input feature img is left or right
    :param method:
    :param k: the conv counts of the first stage, the feature extraction stage
    :param batch_size:
    :param channel:
    :param D:
    :param H:
    :param W:
    :return:
    '''
    origin = input_feature  # batch_size x channel x h // 2 ** k x w // 2 ** k
    candidate = candidate_feature
    # 如果输入的图片是左图, 我们需要去和右图进行比较,
    """ if the input image is the left image, and needs to compare with the right candidate.
        Then it should move to left and pad in right"""
    if position == 'left':
        leftMinusRightMove_List = []
        for disparity in range(D // 2 ** k):   # 192 / (2 ** 4) = 12
            if disparity == 0:
                if method == 'subtract':
                    # 相减
                    leftMinusRightMove = origin - candidate
                else:
                    # 拼接
                    leftMinusRightMove = torch.cat((origin, candidate), 1)  # 拼接

                leftMinusRightMove_List.append(leftMinusRightMove)
            else:
                # 按列进行填充
                zero_padding = np.zeros((origin.shape[0], channel, origin.shape[2], disparity))
                zero_padding = torch.from_numpy(zero_padding).float()
                left_move = torch.cat((origin, zero_padding), 3)

                if method == 'subtract':
                    # 当disparity=2　做法是 将左视图向左平移两个单位, 然后在右边空缺的两列填充零　最后减去右视图
                    leftMinusRightMove = left_move[:, :, :, :origin.shape[3]] - candidate
                else:
                    leftMinusRightMove = torch.cat((left_move[:, :, :, :origin.shape[3]], candidate))

                leftMinusRightMove_List.append(leftMinusRightMove)

        cost_volume = torch.stack(leftMinusRightMove_List, dim=1)
        # print(cost_volume.size())   # torch.Size([2, 12, 32, 16, 32])  # 类似于堆叠12次
        return cost_volume     # batch_size x count(disparity) x channel x H x W

