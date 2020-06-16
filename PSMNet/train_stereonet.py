"""

@file  : train_stereonet.py

@author: xiaolu

@time  : 2020-01-14

"""
import argparse
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from StereoNet import StereoNet
from dataloader.KITTI2015_loader import KITTI2015, RandomCrop, ToTensor, Normalize, Pad
from config import Config


def train(imgL, imgR, disp_L):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL)).to(Config.device)
    imgR = Variable(torch.FloatTensor(imgR)).to(Config.device)
    disp_L = Variable(torch.FloatTensor(disp_L)).to(Config.device)

    mask = disp_L < args.maxdisp
    mask.detach_()

    optimizer.zero_grad()

    output = model(imgL, imgR)

    output = torch.squeeze(output, 1)

    loss = F.smooth_l1_loss(output[mask], disp_L[mask], size_average=True)

    loss.backward()
    optimizer.step()
    return loss.data.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StereoNet')
    parser.add_argument('--maxdisp', type=int, default=192, help='max diparity')
    parser.add_argument('--logdir', default='log/runs', help='log directory')
    parser.add_argument('--datadir', default='./data/KITTI/2015/data_scene_flow', help='data directory')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--validate-batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--log-per-step', type=int, default=1, help='log per step')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='save model per epoch')
    parser.add_argument('--model-dir', default='checkpoint', help='directory where save model checkpoint')
    parser.add_argument('--model-path', default=None, help='path of model to load')
    # parser.add_argument('--start-step', type=int, default=0, help='number of steps at starting')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num-epochs', type=int, default=300, help='number of training epochs')
    parser.add_argument('--num-workers', type=int, default=8, help='num workers in loading data')
    # parser.add_argument('--')

    args = parser.parse_args()

    mean = [0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229]

    train_transform = T.Compose([RandomCrop([256, 512]), Normalize(mean, std), ToTensor()])
    train_dataset = KITTI2015(args.datadir, mode='train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    validate_transform = T.Compose([Normalize(mean, std), ToTensor(), Pad(384, 1248)])
    validate_dataset = KITTI2015(args.datadir, mode='validate', transform=validate_transform)
    validate_loader = DataLoader(validate_dataset, batch_size=args.validate_batch_size, num_workers=args.num_workers)

    cost_volume_method = "subtract"

    model = StereoNet(args.batch_size, cost_volume_method).to(Config.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    Epoch = 2000
    s = 0

    checkpoint_path = './checkpoint/checkpoint_sceneflow.tar'

    for e in range(Epoch):
        avg_train_loss = 0
        for batch in train_loader:
            s += 1
            left = batch['left'].to(Config.device)
            right = batch['right'].to(Config.device)
            disp = batch['disp'].to(Config.device)
            loss = train(left, right, disp)
            print("epoch:{}, step:{}, loss:{}".format(e, s, loss))
            avg_train_loss += loss

            if s % 5 == 0:
                # save
                torch.save({
                    'state_dict': model.state_dict(),
                    'total_train_loss': avg_train_loss,
                    'epoch': e + 1,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)

        print('epoch %d total training loss = %.3f' % (e, avg_train_loss / len(train_loader)))
