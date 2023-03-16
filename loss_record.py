import pdb

import torch
import torch.nn as nn
import os
import argparse
import numpy as np
import torchvision.utils
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug, TensorDataset, epoch
import wandb
import copy
import random
import math
from reparam_module import ReparamModule

def syn_loss_record(trainloader, args):
    '''load synset'''
    pdb.set_trace()
    image_path = os.path.join(args.image_path, args.dataset, args.run_name)
    images_best = []
    labels_best = []
    args.lrs_net = [torch.tensor(eval(lr)).to(args.device).item() for lr in args.lrs_net.split(',')]
    args.snapshots = [torch.tensor(eval(lr)).to(args.device).item() for lr in args.snapshots.split(',')]
    for f in args.files_name.split(','):
        if images_best:
            image_syn = torch.cat([images_best[-1], torch.load(os.path.join(image_path, f, 'images_best.pt'))], dim=0)
            label_syn = torch.cat([labels_best[-1], torch.load(os.path.join(image_path, f, 'labels_best.pt'))], dim=0)
        else:
            image_syn = torch.load(os.path.join(image_path, f, 'images_best.pt'))
            label_syn = torch.load(os.path.join(image_path, f, 'labels_best.pt'))
        if args.dsa and (not args.no_aug):
            DiffAugment(image_syn, args.dsa_strategy, param=args.dsa_param)
        images_best.append(image_syn)
        labels_best.append(label_syn)

    '''training record (syn)'''
    criterion = nn.CrossEntropyLoss().to(args.device)
    criterion_redu_none = nn.CrossEntropyLoss(reduction='none').to(args.device)
    syn_losses = np.zeros((len(trainloader.dataset), sum(args.snapshots)))
    syn_epoch = 0

    for img, lab, lr, num_snapshot in zip(images_best, labels_best, args.lrs_net, args.snapshots):
        eval_labs = lab
        args.lr_net = lr

        with torch.no_grad():
            image_save = img
        image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(
            eval_labs.detach())  # avoid any unaware modification

        net_eval = get_network(args.model, args.channel, args.num_classes, args.im_size).to(args.device)  # get a random model
        lr = float(args.lr_net)
        Epoch = int(args.epoch_eval_train)
        epoch_test_pool = list(range(0, Epoch, math.ceil(Epoch/num_snapshot)))
        lr_schedule = [Epoch//2+1]
        optimizer = torch.optim.SGD(net_eval.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

        dst_eval = TensorDataset(copy.deepcopy(image_syn_eval.detach()), copy.deepcopy(label_syn_eval.detach()))
        evalloader = torch.utils.data.DataLoader(dst_eval, batch_size=args.batch_train, shuffle=False, num_workers=0)


        for ep in tqdm(range(Epoch+1)):
            loss_train, acc_train = epoch('train', evalloader, net_eval, optimizer, criterion, args, aug=True, texture=args.texture)
            if ep in epoch_test_pool:
                with torch.no_grad():
                    losses = epoch('test', trainloader, net_eval, optimizer, criterion_redu_none, args, aug=False, record=True)
                    syn_losses[:, syn_epoch] = losses
            if ep in lr_schedule:
                lr *= 0.1
                optimizer = torch.optim.SGD(net_eval.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    torch.save(syn_losses, os.path.join(image_path, 'training_loss.pt'))
    return syn_losses

def ori_loss_record(trainloader, args):

    expert_dir = args.buffer_path
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)

    print("Expert Dir: {}".format(expert_dir))
    buffer = [_ for _ in os.listdir(expert_dir) if 'replay_buffer_' in _]
    random.shuffle(buffer)
    r_buffer = buffer[0].split('.')[0].split('_')[-1]
    r_expert = np.random.randint(0, len(buffer[0]))
    expert_trajectory = torch.load(os.path.join(expert_dir, buffer[0]))[r_expert]

    ''' initialize expert model'''
    criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
    teacher_net = get_network(args.model, args.channel, args.num_classes, args.im_size).to(args.device)
    teacher_net = ReparamModule(teacher_net).to(args.device)



    '''training record (ori)'''
    ori_losses = np.zeros((len(trainloader.dataset), len(expert_trajectory)))
    for i, forward_params in enumerate(expert_trajectory):
        forward_params = torch.cat([p.data.to(args.device).reshape(-1) for p in forward_params], 0)
        losses = epoch('test', trainloader, teacher_net, None, criterion, args, aug=False, flat_parameter=forward_params, record=True)
        ori_losses[:,i] = losses
    pdb.set_trace()
    torch.save(ori_losses, os.path.join(expert_dir, f'training_loss_origin_buffer{r_buffer}_expert{r_expert}.pt'))
    return ori_losses


def main(args):
    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")

    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")


    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args, record=True)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=False, num_workers=0)

    im_res = im_size[0]

    args.im_size = im_size
    args.channel = channel
    args.num_classes = num_classes


    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None


    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    # if args.batch_syn is None:
    #     args.batch_syn = num_classes * args.ipc

    args.distributed = torch.cuda.device_count() > 1


    print('Hyper-parameters: \n', args.__dict__)


    if args.record_type == "origin":
        losses = ori_loss_record(trainloader, args)
    elif args.record_type == "synset":
        losses = syn_loss_record(trainloader, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--save_dir', type=str, default='./logged_files', help='save path')
    parser.add_argument('--image_path', type=str, default='./logged_files', help='image path')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--run_name', type=str, default=None, help="run_name")
    parser.add_argument('--files_name', type=str, default=None, help="file_name(epoch)")
    parser.add_argument('--lrs_net', type=str, default=None, help="lrs_net")
    parser.add_argument('--snapshots', type=str, default=None, help="the number of snapshots in every synset, the sum should be 50")


    parser.add_argument('--record_type', type=str, default='origin', help="origin or synset")
    args = parser.parse_args()

    main(args)


