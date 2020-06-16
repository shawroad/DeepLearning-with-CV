"""

@file  : inference.py

@author: xiaolu

@time  : 2020-03-06

"""
import argparse
import cv2
import torch
import torch.nn.functional as F
from dataloader.KITTI2015_loader import KITTI2015, RandomCrop, ToTensor, Normalize, Pad
from model import StereoNet
import torchvision.transforms as T

left_path = './data/KITTI/2015/data_scene_flow/training/image_2/000001_10.png'
right_path = './data/KITTI/2015/data_scene_flow/training/image_3/000001_10.png'
disp_path = './data/KITTI/2015/data_scene_flow/training/disp_occ_0/000001_10.png'

parser = argparse.ArgumentParser(description='PSMNet inference')
parser.add_argument('--left', default=left_path, help='path to the left image')
parser.add_argument('--right', default=right_path, help='path to the right image')
parser.add_argument('--disp', default=disp_path, help='path to the disp image')
parser.add_argument('--model_path', default='./kitti15.tar', help='path to the model')
parser.add_argument('--save-path', default='s_img.png', help='path to save the disp image')
args = parser.parse_args()


mean = [0.406, 0.456, 0.485]
std = [0.225, 0.224, 0.229]
device_ids = [0, 1, 2, 3]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    '''
    测试
    :return:
    '''

    # model_state = 'StereoNet.pkl'
    left = cv2.imread(args.left)
    right = cv2.imread(args.right)
    disp_l = cv2.imread(args.disp)

    pairs = {'left': left, 'right': right, 'disp': disp_l}

    transform = T.Compose([Normalize(mean, std), ToTensor(), Pad(384, 1248)])
    pairs = transform(pairs)
    left = pairs['left'].to(device).unsqueeze(0)
    right = pairs['right'].to(device).unsqueeze(0)
    disp_l = pairs['disp'].to(device).unsqueeze(0)

    cost_volume_method = "subtract"

    model = StereoNet(1, cost_volume_method=cost_volume_method)

    state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['state_dict'])

    model.eval()
    print("模型加载成功")
    print("正在预测中ing...")
    with torch.no_grad():
        output = model(left, right)
    output = output.view(output.size(2), output.size(3), -1)
    cv2.imwrite('131313.png', output.numpy())


if __name__ == '__main__':
    main()
