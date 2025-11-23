import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
from tensorboardX import SummaryWriter
from src.model.networks.deeplab3d_multi import DeeplabMulti
from src.model.networks.meta_deeplab_multi import Res_Deeplab
from src.model.networks.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d

import datetime
import time

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def build_model(args):

    net = Res_Deeplab(num_classes=args.num_classes)
    #print(net)

    if torch.cuda.is_available():
        net.cuda()
        torch.backends.cudnn.benchmark=True

    return net

def to_var(x, requires_grad=True):
    x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d(is_softmax=False).cuda()

    return criterion(pred, label)

def obtain_meta(source_img):
    seg_model = DeeplabMulti(num_classes=19).cuda()
    seg_model.load_state_dict(torch.load('/home/cyang53/CED/Baseline/AdaptSegNet-CVPR2018/snapshots/GTA5_best.pth'))
    dis_model = FCDiscriminator(num_classes=19).cuda()
    dis_model.load_state_dict(torch.load('/home/cyang53/CED/Ours/AdaptSegNet-CVPR2018/snapshots/GTA5_best_D2.pth'))
    seg_model.eval()
    dis_model.eval()

    output1, output2 = seg_model(source_img)
    meta_map = dis_model(F.softmax(output2, dim=1)).cpu().data[0]
    source_like = torch.where(meta_map < 0.5)
    return source_like

def main():
    """Create the model and start the training."""
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.log_dir + '/result'):
        os.makedirs(args.log_dir + '/result')

    best_mIoU = 0
    mIoU = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    metaloader = data.DataLoader(GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
    crop_size=input_size_target, scale=False, mirror=args.random_mirror, mean=IMG_MEAN), batch_size=args.update_f * args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    targetloader = data.DataLoader(cityscapesPseudo(args.data_dir_target, args.data_list_target,
    max_iters=args.num_steps * args.iter_size * args.batch_size,
    crop_size=input_size_target,
    scale=False, mirror=args.random_mirror, mean=IMG_MEAN),
    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    pin_memory=True)

    targetloader_iter = enumerate(targetloader)

    main_model = build_model(args)
    saved_state_dict = torch.load(args.restore_from)
    pretrained_dict = {k:v for k,v in saved_state_dict.items() if k in main_model.state_dict()}
    main_model.load_state_dict(pretrained_dict)

    optimizer = optim.SGD(main_model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()



    interp = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)


    # set up tensor board
    if args.tensorboard:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        writer = SummaryWriter(args.log_dir)

    for i_iter in range(args.num_steps):
        if args.is_meta:
            main_model.train()
            l_f_meta = 0
            l_g_meta = 0
            l_f = 0

            optimizer.zero_grad()
            adjust_learning_rate(optimizer, i_iter)

            meta_net = Res_Deeplab(num_classes=args.num_classes)
            meta_net.load_state_dict(main_model.state_dict())
            meta_net.cuda()

            _, batch = targetloader_iter.__next__()
            image, label, _, _ = batch

            image = to_var(image, requires_grad=False)
            label = to_var(label, requires_grad=False)

            T1 = to_var(torch.eye(19, 19))
            T2 = to_var(torch.eye(19, 19))
            
            #metanet update
            y_f_hat1, y_f_hat2 = meta_net(image)
            y_f_hat1 = torch.softmax(interp_target(y_f_hat1), dim=1).permute(0, 2, 3, 1).contiguous().view(-1, args.num_classes)
            y_f_hat2 = torch.softmax(interp_target(y_f_hat2), dim=1).permute(0, 2, 3, 1).contiguous().view(-1, args.num_classes)

            pre1 = torch.mm(y_f_hat1, T1).view(args.batch_size, h, w, args.num_classes).permute(0, 3, 1, 2)
            pre2 = torch.mm(y_f_hat2, T2).view(args.batch_size, h, w, args.num_classes).permute(0, 3, 1, 2)
            l_f_meta = loss_calc(pre2, label) + 0.1 * loss_calc(pre1, label)

            meta_net.zero_grad()

            grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
            meta_net.update_params(1e-3, source_params=grads)

            x_val, y_val, _, _ = next(iter(metaloader))
            x_val = to_var(x_val, requires_grad=False)
            y_val = to_var(y_val, requires_grad=False)
            meta_source = obtain_meta(x_val)
            y_val[meta_source] = 255

            y_g_hat1, y_g_hat2 = meta_net(x_val)
            y_g_hat1 = torch.softmax(interp(y_g_hat1), dim=1)
            y_g_hat2 = torch.softmax(interp(y_g_hat2), dim=1)

            l_g_meta = loss_calc(y_g_hat2, y_val) + 0.1 * loss_calc(y_g_hat1, y_val)
            grad_eps1 = torch.autograd.grad(l_g_meta, T1, only_inputs=True, retain_graph=True)[0]
            grad_eps2 = torch.autograd.grad(l_g_meta, T2, only_inputs=True)[0]

            grad_eps1 = grad_eps1 / torch.max(grad_eps1)
            T1 = torch.clamp(T1-0.11*grad_eps1,min=0)
            # T1 = torch.softmax(T1, 1)
            norm_c = torch.sum(T1, 1)

            for j in range(args.num_classes):
                if norm_c[j] != 0:
                    T1[j, :] /= norm_c[j]

            grad_eps2 = grad_eps2 / torch.max(grad_eps2)
            T2 = torch.clamp(T2-0.11*grad_eps2,min=0)

            norm_c = torch.sum(T2, 1)


            for j in range(args.num_classes):
                if norm_c[j] != 0:
                    T2[j, :] /= norm_c[j]

            #segmentation net update
            y_f_hat1, y_f_hat2 = main_model(image)
            y_f_hat1 = torch.softmax(interp_target(y_f_hat1), dim=1).permute(0, 2, 3, 1).contiguous().view(-1, args.num_classes)
            y_f_hat2 = torch.softmax(interp_target(y_f_hat2), dim=1).permute(0, 2, 3, 1).contiguous().view(-1, args.num_classes)
            pre1 = torch.mm(y_f_hat1, T1).view(args.batch_size, h, w, args.num_classes).permute(0, 3, 1, 2)
            pre2 = torch.mm(y_f_hat2, T2).view(args.batch_size, h, w, args.num_classes).permute(0, 3, 1, 2)

            l_f = loss_calc(pre2, label) + 0.1 * loss_calc(pre1, label)
            optimizer.zero_grad()
            l_f.backward()
            optimizer.step()

            if args.tensorboard:
                scalar_info = {
                    'loss_g_meta': l_g_meta.item(),
                    'loss_f_meta': l_f_meta.item(),
                    'loss_f': l_f.item(),
                }

                if i_iter % 10 == 0:
                    for key, val in scalar_info.items():
                        writer.add_scalar(key, val, i_iter)

            print('exp = {}'.format(args.log_dir))
            print(
            'iter = {0:8d}/{1:8d}, loss_g_meta = {2:.3f} loss_f_meta = {3:.3f} loss_f = {4:.3f}'.format(
                i_iter, args.num_steps, l_g_meta.item(), l_f_meta.item(), l_f.item()))

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(main_model.state_dict(), osp.join(args.log_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            break
        if i_iter % args.save_pred_every == 0 and i_iter > 0:
            now = datetime.datetime.now()
            print (now.strftime("%Y-%m-%d %H:%M:%S"), '  Begin evaluation on iter {0:8d}/{1:8d}  '.format(i_iter, args.num_steps))
            mIoU = evaluate(main_model, pred_dir=args.log_dir + '/result')
            writer.add_scalar('mIoU', mIoU, i_iter)
            print('Finish Evaluation: '+time.asctime(time.localtime(time.time())))
            if mIoU > best_mIoU:
                best_mIoU = mIoU
                torch.save(main_model.state_dict(), osp.join(args.log_dir, 'MetaCorrection_best.pth'))

    if args.tensorboard:
        writer.close()