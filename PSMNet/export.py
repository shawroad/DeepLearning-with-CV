"""

@file  : export.py

@author: xiaolu

@time  : 2020-01-14

"""
import time
import torch
from StereoNet import StereoNet


if __name__ == '__main__':
    '''
    从tar中提取模型 整理成pt文件
    '''
    checkpoint = './checkpoint/checkpoint_sceneflow.tar'

    print('loading {}...'.format(checkpoint))
    start = time.time()
    checkpoint = torch.load(checkpoint)
    print('elapsed {} sec'.format(time.time() - start))
    model = checkpoint['state_dict']
    print(type(model))

    filename = 'ScereoNet_model.pt'
    print('saving {}...'.format(filename))
    start = time.time()
    torch.save(model, filename)
    print('elapsed {} sec'.format(time.time() - start))

    print('loading {}...'.format(filename))
    start = time.time()
    model = StereoNet(batch_size=1, cost_volume_method="subtract")
    model.load_state_dict(torch.load(filename))
    print('elapsed {} sec'.format(time.time() - start))
