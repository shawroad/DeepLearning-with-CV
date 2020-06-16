"""

@file  : StereoNet.py

@author: xiaolu

@time  : 2020-01-14

"""
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from config import Config


class StereoNet(nn.Module):
    def __init__(self, batch_size, cost_volume_method):
        super(StereoNet, self).__init__()

        self.batch_size = batch_size
        self.cost_volume_method = cost_volume_method

        cost_volume_channel = 32

        if cost_volume_method == 'subtract':
            cost_volume_channel = 32
        elif cost_volume_method == 'concat':
            cost_volume_channel = 64

        else:
            print("没有你指定{}的方法".format(cost_volume_method))

        # 下采样
        self.downsampling = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.Conv2d(32, 32, 5, stride=2, padding=2),
            nn.Conv2d(32, 32, 5, stride=2, padding=2),
            nn.Conv2d(32, 32, 5, stride=2, padding=2),
        )

        # 多个残差块的拼接
        self.res = nn.Sequential(
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            nn.Conv2d(32, 32, 3, 1, 1),
        )

        # using 3d conv to instead the Euclidean distance
        self.cost_volume_filter = nn.Sequential(
            MetricBlock(cost_volume_channel, 32),
            MetricBlock(32, 32),
            MetricBlock(32, 32),
            MetricBlock(32, 32),
            nn.Conv3d(32, 1, 3, padding=1),   # 保证了最后的通道数整成1
        )

        self.refine = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),
            ResBlock(32, 32, dilation=1),
            ResBlock(32, 32, dilation=2),
            ResBlock(32, 32, dilation=4),
            ResBlock(32, 32, dilation=8),
            ResBlock(32, 32, dilation=1),
            ResBlock(32, 32, dilation=1),
            nn.Conv2d(32, 1, 3, padding=1),
            # nn.ReLU(),
        )

    def forward_once_1(self, x):
        # 下采样加残差快
        output = self.downsampling(x)
        output = self.res(output)
        return output

    def forward_stage1(self, input_l, input_r):
        # 对左右图片进行第一阶段的下采样和残差　也就是第一波提取特征
        output_l = self.forward_once_1(input_l)
        output_r = self.forward_once_1(input_r)
        return output_l, output_r

    def forward_once_2(self, cost_volume):
        # 过滤视差?
        cost_volume = cost_volume.permute([0, 2, 1, 3, 4])
        output = self.cost_volume_filter(cost_volume)  # [batch_size, channel, disparity, h, w]
        disparity_low = output
        return disparity_low   # low resolution disparity map

    def forward_stage2(self, feature_l, feature_r):
        # 计算代价网络
        cost_v_l = CostVolume(feature_l, feature_r, "left", method=self.cost_volume_method, k=4, batch_size=self.batch_size)
        disparity_low = self.forward_once_2((cost_v_l))
        # print(disparity_low.size())   # torch.Size([2, 1, 12, 16, 32])  # 通道数搞成1
        disparity_low = torch.squeeze(disparity_low, dim=1)
        return disparity_low

    def forward_stage3(self, disparity_low, left):
        # 将第二个阶段的特征图　进行上采样　然后进行像素注意力　最后与左视图进行拼接
        d_high = F.interpolate(disparity_low, [left.shape[2], left.shape[3]], mode='bilinear', align_corners=True)
        # print(d_high.size())   # torch.Size([2, 12, 256, 512])

        d_high = soft_argmin(d_high)
        d_concat = torch.cat([d_high, left], dim=1)
        # print(d_concat.size())

        d_refined = self.refine(d_concat)
        print(d_refined.size())   # torch.Size([2, 1, 256, 512])

        return d_refined

    def forward(self, left, right):
        # 第一波特征提取
        # 原始的left, right大小为torch.Size([2, 3, 256, 512])
        left_feature, right_feature = self.forward_stage1(left, right)
        # print(left_feature.size())    # torch.Size([2, 32, 16, 32])  长宽缩短为原来的1/4

        disparity_low_l = self.forward_stage2(left_feature, right_feature)
        # print(disparity_low_l.size())   # torch.Size([2, 12, 16, 32])

        d_initial_l = F.interpolate(disparity_low_l, [left.shape[2], left.shape[3]], mode='bilinear', align_corners=True)
        # print(d_initial_l.size())   # torch.Size([2, 12, 256, 512]) 用线性差值的方式恢复成原来的维度
        d_initial_l = soft_argmin(d_initial_l)
        # print(d_initial_l.size())   # torch.Size([2, 1, 256, 512])

        d_refined_l = self.forward_stage3(disparity_low_l, left)

        d_final_l = d_initial_l + d_refined_l
        # d_final_l = soft_argmin(d_initial_l + d_refined_l)

        d_final_l = nn.ReLU()(d_final_l)
        return d_final_l


class MetricBlock(nn.Module):
    '''
    三位卷积　+ 批量归一化　+ 激活
    '''
    def __init__(self, in_channel, out_channel, stride=1):
        super(MetricBlock, self).__init__()
        self.conv3d_1 = nn.Conv3d(in_channel, out_channel, 3, 1, 1)  # kernel_size,  stride, padding
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv3d_1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    # 类似进行两次空洞卷积　然后两步运算完的结果　引入输出的残差
    def __init__(self, in_channel, out_channel, dilation=1, stride=1, downsample=None):
        super(ResBlock, self).__init__()

        padding = dilation

        # 第一波卷积
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=padding,
                               dilation=dilation, bias=False)   # 空洞卷积
        self.bn1 = nn.BatchNorm2d(out_channel)  # 批量归一化
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)  # 激活函数

        # 第二波卷积
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=padding,
                               dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # 下采样
        self.downsample = downsample
        self.stride = stride
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.p = padding
        self.d = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual

        out = self.relu2(out)

        return out


def soft_argmin(cost_volume):
    # 按第二维度进行softmax 然后得出每个像素点的权重　像素值乘权重累加　最后就将图片拍成一维
    '''
    Remove single-dimensional entries from the shape of an array.
    :param cost_volume:
    :return:
    '''
    softmax = nn.Softmax(dim=1)
    disparity_softmax = softmax(-cost_volume)
    # print(disparity_softmax.size())  # torch.Size([2, 12, 256, 512])

    d_grid = torch.arange(cost_volume.shape[1], dtype=torch.float)
    d_grid = d_grid.reshape(-1, 1, 1)
    d_grid = d_grid.repeat((cost_volume.shape[0], 1, cost_volume.shape[2], cost_volume.shape[3]))
    # print(d_grid.size())   # torch.Size([2, 12, 256, 512])

    d_grid = d_grid.to(Config.device)

    tmp = disparity_softmax * d_grid

    arg_soft_min = torch.sum(tmp, dim=1, keepdim=True)
    # print(arg_soft_min.size())   # torch.Size([2, 1, 256, 512])
    return arg_soft_min


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

