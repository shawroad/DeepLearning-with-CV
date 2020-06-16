"""

@file  : config.py

@author: xiaolu

@time  : 2020-01-16

"""
import torch


class Config:
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
