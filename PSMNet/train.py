"""

@file  : train_RNet.py

@author: xiaolu

@time  : 2019-11-13

"""
import argparse
import os
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from models.PSMnet import PSMNet
from models.smoothloss import SmoothL1Loss
from dataloader.KITTI2015_loader import KITTI2015, RandomCrop, ToTensor, Normalize, Pad
import tensorboardX as tX
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')


parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=192, help='max diparity')
parser.add_argument('--logdir', default='log/runs', help='log directory')
parser.add_argument('--datadir', default='./data/KITTI/2015/data_scene_flow', help='data directory')
parser.add_argument('--cuda', type=int, default=0, help='gpu number')
parser.add_argument('--batch-size', type=int, default=1, help='batch size')
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
device_ids = [0, 1, 2, 3]   # 可用的gpu个数

writer = tX.SummaryWriter(log_dir=args.logdir, comment='FSMNet')   # 存日志
# device = torch.device('cuda')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)


def main(args):
    # 1. 加载训练集, 验证集数据
    train_transform = T.Compose([RandomCrop([256, 512]), Normalize(mean, std), ToTensor()])
    train_dataset = KITTI2015(args.datadir, mode='train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    validate_transform = T.Compose([Normalize(mean, std), ToTensor(), Pad(384, 1248)])
    validate_dataset = KITTI2015(args.datadir, mode='validate', transform=validate_transform)
    validate_loader = DataLoader(validate_dataset, batch_size=args.validate_batch_size, num_workers=args.num_workers)

    step = 0
    best_error = 100.0

    model = PSMNet(args.maxdisp).to(device)
    # model = nn.DataParallel(model, device_ids=device_ids)  # 多gpu运行

    criterion = SmoothL1Loss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.model_path is not None:
        # 如果模型路径不空 我们就加载模型
        state = torch.load(args.model_path)
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        step = state['step']
        best_error = state['error']
        print('load model from {}'.format(args.model_path))

    # 打印出模型的参数了量
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # 开始训练
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        step = train(model, train_loader, optimizer, criterion, step)
        adjust_lr(optimizer, epoch)  # 不断进行学习率的调整

        if epoch % args.save_per_epoch == 0:
            model.eval()
            error = validate(model, validate_loader, epoch)
            best_error = save(model, optimizer, epoch, step, error, best_error)


def validate(model, validate_loader, epoch):
    '''
    验证　validate 40 image pairs
    :param model:
    :param validate_loader:
    :param epoch:
    :return:
    '''
    num_batches = len(validate_loader)
    idx = np.random.randint(num_batches)

    avg_error = 0.0
    for i, batch in enumerate(validate_loader):
        left_img = batch['left'].to(device)
        right_img = batch['right'].to(device)
        target_disp = batch['disp'].to(device)

        mask = (target_disp > 0)
        mask = mask.detach_()

        with torch.no_grad():
            _, _, disp = model(left_img, right_img)

        delta = torch.abs(disp[mask] - target_disp[mask])
        error_mat = (((delta >= 3.0) + (delta >= 0.05 * (target_disp[mask]))) == 2)
        error = torch.sum(error_mat).item() / torch.numel(disp[mask]) * 100

        avg_error += error
        if i == idx:
            left_save = left_img
            disp_save = disp

    avg_error = avg_error / num_batches
    print('epoch: {:03} | 3px-error: {:.5}%'.format(epoch, avg_error))
    writer.add_scalar('error/3px', avg_error, epoch)
    save_image(left_save[0], disp_save[0], epoch)    # 保存此图片
    return avg_error


def save_image(left_image, disp, epoch):
    for i in range(3):
        left_image[i] = left_image[i] * std[i] + mean[i]
    b, r = left_image[0], left_image[2]
    left_image[0] = r  # BGR --> RGB
    left_image[2] = b
    # left_image = torch.from_numpy(left_image.cpu().numpy()[::-1])

    disp_img = disp.detach().cpu().numpy()
    fig = plt.figure(figsize=(12.84, 3.84))
    plt.axis('off')  # hide axis
    plt.imshow(disp_img)
    plt.colorbar()

    writer.add_figure('image/disp', fig, global_step=epoch)
    writer.add_image('image/left', left_image, global_step=epoch)


def train(model, train_loader, optimizer, criterion, step):
    '''
    训练一个epoch
    :param model:
    :param train_loader:
    :param optimizer:
    :param criterion:
    :param step:
    :return:
    '''
    for batch in train_loader:
        step += 1
        optimizer.zero_grad()

        left_img = batch['left'].to(device)
        right_img = batch['right'].to(device)
        target_disp = batch['disp'].to(device)
        print(left_img.size())
        print(right_img.size())
        print(target_disp.size())

        mask = (target_disp > 0)
        mask = mask.detach_()

        # 得到输出并进行损失的计算
        disp1, disp2, disp3 = model(left_img, right_img)
        loss1, loss2, loss3 = criterion(disp1[mask], disp2[mask], disp3[mask], target_disp[mask])
        # 本文作者损失给出了不同的权重
        total_loss = 0.5 * loss1 + 0.7 * loss2 + 1.0 * loss3

        total_loss.backward()
        optimizer.step()

        if step % args.log_per_step == 0:  # 多少步进行损失的记录
            writer.add_scalar('loss/loss1', loss1, step)
            writer.add_scalar('loss/loss2', loss2, step)
            writer.add_scalar('loss/loss3', loss3, step)
            writer.add_scalar('loss/total_loss', total_loss, step)
            print('step: {:05} | total loss: {:.5} | loss1: {:.5} | loss2: {:.5} | loss3: {:.5}'.format(step, total_loss.item(), loss1.item(), loss2.item(), loss3.item()))
    return step


def adjust_lr(optimizer, epoch):
    '''
    不断调整学习率
    :param optimizer:
    :param epoch:
    :return:
    '''
    if epoch == 200:
        lr = 0.0001
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def save(model, optimizer, epoch, step, error, best_error):
    '''
    保存模型
    :param model:
    :param optimizer:
    :param epoch:
    :param step:
    :param error:
    :param best_error:
    :return:
    '''
    path = os.path.join(args.model_dir, '{:03}.ckpt'.format(epoch))  # 名字叫做path
    # 普通办法的保存模型
    # torch.save(model.state_dict(), path)
    # model.save_state_dict(path)

    # 另外一种保存 定制我们要保存的信息
    state = {}
    state['state_dict'] = model.state_dict()
    state['optimizer'] = optimizer.state_dict()
    state['error'] = error
    state['epoch'] = epoch
    state['step'] = step

    torch.save(state, path)
    print('save model at epoch{}'.format(epoch))

    if error < best_error:
        best_error = error
        best_path = os.path.join(args.model_dir, 'best_model.ckpt'.format(epoch))
        shutil.copyfile(path, best_path)  # best_model始终保持最好的模型
        print('best model in epoch {}'.format(epoch))

    return best_error


if __name__ == '__main__':
    main(args)
    writer.close()
