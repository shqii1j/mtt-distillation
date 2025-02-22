import os
import argparse
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import wandb
import copy
import random
import math
from reparam_module import ReparamModule
torch.cuda.empty_cache()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):
    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")

    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    im_res = im_size[0]

    args.im_size = im_size

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []


    if args.dsa:
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None

    wandb.init(sync_tensorboard=False,
               project="DatasetDistillation",
               job_type="CleanRepo",
               config=args,
               )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    args.distributed = torch.cuda.device_count() > 1


    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    ''' Organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        images_all.append(torch.unsqueeze(sample[0], dim=0))
        labels_all.append(class_map[torch.tensor(sample[1]).item()])

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    ''' Initialize expert file'''
    expert_dir = args.buffer_path
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset)
    if args.dataset in ["CIFAR10", "CIFAR100", "SVHN"] and not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)

    print("Expert Dir: {}".format(expert_dir))


    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)


    ''' *Load previous subsets'''
    if args.reparam_syn:
        images_best = []
        labels_best = []
        if not args.lrs_net:
            args.lrs_net = []
        else:
            args.lrs_net = [torch.tensor(eval(lr)).to(args.device).item() for lr in args.lrs_net.split(',')]

        image_path = os.path.join(args.image_path, args.dataset, args.run_name)
        for f in args.pre_names.split(','):
            if images_best:
                img = torch.cat([images_best[-1], torch.load(os.path.join(image_path, f, 'images_best.pt')).to(args.device).requires_grad_(False)], dim=0)
                lab = torch.cat([labels_best[-1], torch.load(os.path.join(image_path, f, 'labels_best.pt')).to(args.device)], dim=0)
            else:
                img = torch.load(os.path.join(image_path, f, 'images_best.pt')).to(args.device).requires_grad_(False)
                lab = torch.load(os.path.join(image_path, f, 'labels_best.pt')).to(args.device)
            images_best.append(img)
            labels_best.append(lab)

            if not args.lrs_net:
                args.lrs_net.append(torch.load(os.path.join(image_path, f, 'best_lr.pt')))


    ''' *Distill '''
    intervals = [(eval(i.split('-')[0]), eval(i.split('-')[1])) for i in args.intervals.split(",")]
    print(f'Intervals: {intervals}')
    if args.ipcs:
        ipcs = [eval(_) for _ in args.ipcs.split(',')]
    else:
        ipcs = [args.ipc] * len(intervals)

    for k, inter in enumerate(intervals):
        start_epoch = inter[0]
        end_epoch = inter[1]
        ipc = ipcs[k]
        if args.batch_syn == None:
            batch_syn = num_classes * ipc
        else:
            batch_syn = args.batch_syn
        print(f'\n\n\nTraining epoch from {start_epoch} to {end_epoch}, with {ipc} ipc')

        ''' initialize the synthetic data '''
        label_syn = torch.tensor([np.ones(ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        if args.texture:
            image_syn = torch.randn(size=(num_classes * ipc, channel, im_size[0]*args.canvas_size, im_size[1]*args.canvas_size), dtype=torch.float)
        else:
            image_syn = torch.randn(size=(num_classes * ipc, channel, im_size[0], im_size[1]), dtype=torch.float)

        if args.pix_init == 'real':
            print('initialize synthetic data from random real images')
            if args.texture:
                for c in range(num_classes):
                    for i in range(args.canvas_size):
                        for j in range(args.canvas_size):
                            image_syn.data[c * ipc:(c + 1) * ipc, :, i * im_size[0]:(i + 1) * im_size[0],
                            j * im_size[1]:(j + 1) * im_size[1]] = torch.cat(
                                [get_images(c, 1).detach().data for s in range(ipc)])
            for c in range(num_classes):
                image_syn.data[c * ipc:(c + 1) * ipc] = get_images(c, ipc).detach().data
        else:
            print('initialize synthetic data from random noise')


        ''' training '''
        image_syn = image_syn.detach().to(args.device).requires_grad_(True)
        syn_lr = torch.tensor(args.lr_teacher).to(args.device)
        syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
        optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
        optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)
        optimizer_img.zero_grad()

        criterion = nn.CrossEntropyLoss().to(args.device)
        print('%s training begins'%get_time())

        best_acc = {m: 0 for m in model_eval_pool}
        best_std = {m: 0 for m in model_eval_pool}

        seg_path = args.name

        for it in range(0, args.Iteration+1):
            save_this_it = False
            commit = False
            if it == args.Iteration // 1000 * 1000:
                commit = True

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                    if args.dsa:
                        print('DSA augmentation strategy: \n', args.dsa_strategy)
                        print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                    else:
                        print('DC augmentation parameters: \n', args.dc_aug_param)

                    accs_test = []
                    accs_train = []

                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
                        if args.reparam_syn:
                            for img, lab, lr in zip(images_best, labels_best, args.lrs_net):
                                eval_labs = lab
                                args.lr_net = lr
                                with torch.no_grad():
                                    image_save = img
                                image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(
                                    eval_labs.detach())  # avoid any unaware modification
                                net_eval, acc_train, acc_test = evaluate_synset(it, net_eval, image_syn_eval,
                                                                                label_syn_eval,
                                                                                testloader, args,
                                                                                texture=args.texture, printer=True)
                            eval_labs = torch.cat([labels_best[-1], label_syn], 0)
                            with torch.no_grad():
                                image_save = torch.cat([images_best[-1], image_syn], 0)
                        else:
                            eval_labs = label_syn
                            with torch.no_grad():
                                image_save = image_syn

                        image_syn_eval = copy.deepcopy(image_save.detach())
                        label_syn_eval = copy.deepcopy(eval_labs.detach())
                        args.lr_net = syn_lr.item()
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, texture=args.texture, printer=True)
                        accs_test.append(acc_test)
                        accs_train.append(acc_train)
                        del net_eval, image_syn_eval, label_syn_eval
                    accs_test = np.array(accs_test)
                    accs_train = np.array(accs_train)
                    acc_test_mean = np.mean(accs_test)
                    acc_test_std = np.std(accs_test)
                    if acc_test_mean > best_acc[model_eval]:
                        best_acc[model_eval] = acc_test_mean
                        best_std[model_eval] = acc_test_std
                        save_this_it = True
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
                    wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, commit=commit)  # modi
                    wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, commit=commit)  # modi

            if it in eval_it_pool and (save_this_it or it % 1000 == 0):
                with torch.no_grad():
                    image_save = image_syn.cuda()

                    save_dir = os.path.join(args.save_dir, args.dataset, args.run_name, seg_path)        # modi

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
                    torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_{}.pt".format(it)))

                    if save_this_it:
                        torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt"))
                        torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_best.pt"))

                        best_syn_lr = args.lr_net
                        torch.save(torch.tensor(best_syn_lr), os.path.join(save_dir, "best_lr.pt"))
                        wandb.log({'Best_Syn_Lr/{}'.format(model_eval): best_syn_lr}, commit=commit)

                    wandb.log({seg_path+"/Pixels": wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, commit=commit)     # modi

                    if ipc < 50 or args.force_save:
                        upsampled = image_save
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({seg_path+"/Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, commit=commit)        # modi
                        wandb.log({seg_path+'/Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, commit=commit)      # modi

                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std)
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                            wandb.log({seg_path+"/Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, commit=commit)    # modi

                        if args.zca:
                            image_save_zca = image_save.to(args.device)
                            image_save_zca = args.zca_trans.inverse_transform(image_save_zca)
                            image_save_zca.cpu()

                            torch.save(image_save_zca.cpu(), os.path.join(save_dir, "images_zca_{}.pt".format(it)))

                            upsampled = image_save_zca
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                            wandb.log({seg_path+"/Reconstructed_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, commit=commit)        # modi
                            wandb.log({seg_path+'/Reconstructed_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, commit=commit)      # modi

                            for clip_val in [2.5]:
                                std = torch.std(image_save_zca)
                                mean = torch.mean(image_save_zca)
                                upsampled = torch.clip(image_save_zca, min=mean - clip_val * std, max=mean + clip_val * std)
                                if args.dataset != "ImageNet":
                                    upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                    upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                                grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                                wandb.log({seg_path+"/Clipped_Reconstructed_Images/std_{}".format(clip_val): wandb.Image(
                                    torch.nan_to_num(grid.detach().cpu()))}, commit=commit)     # modi


            wandb.log({seg_path+"/Synthetic_LR": syn_lr.detach().cpu()}, commit=commit)     # modi


            ''' *Initialize student network'''
            student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model
            student_net = ReparamModule(student_net)

            if args.distributed:
                student_net = torch.nn.DataParallel(student_net)

            student_net.train()

            num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

            if args.load_all:
                expert_trajectory = buffer[np.random.randint(0, len(buffer))]
            else:
                expert_trajectory = buffer[expert_idx]
                expert_idx += 1
                if expert_idx == len(buffer):
                    expert_idx = 0
                    file_idx += 1
                    if file_idx == len(expert_files):
                        file_idx = 0
                        random.shuffle(expert_files)
                    print("loading file {}".format(expert_files[file_idx]))
                    if args.max_files != 1:
                        del buffer
                        buffer = torch.load(expert_files[file_idx])
                    if args.max_experts is not None:
                        buffer = buffer[:args.max_experts]
                    random.shuffle(buffer)

            rn_start_epoch = np.random.randint(start_epoch, end_epoch - args.expert_epochs)
            starting_params = expert_trajectory[rn_start_epoch]

            target_params = expert_trajectory[rn_start_epoch+args.expert_epochs]       # modi
            target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

            student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]
            starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

            if args.reparam_syn:
                syn_images = torch.concat([images_best[-1], image_syn], dim=0)
                y_hat = torch.concat([labels_best[-1], label_syn.to(args.device)], dim=0)
            else:
                syn_images = image_syn
                y_hat = label_syn.to(args.device)

            param_loss_list = []
            param_dist_list = []
            indices_chunks = []

            for step in range(args.syn_steps):
                ''' *update student network'''
                if not indices_chunks:
                    indices = torch.randperm(len(syn_images))
                    indices_chunks = list(torch.split(indices, batch_syn))

                these_indices = indices_chunks.pop()

                x = syn_images[these_indices]
                this_y = y_hat[these_indices]

                if args.texture:
                    x = torch.cat([torch.stack([torch.roll(im, (torch.randint(im_size[0]*args.canvas_size, (1,)), torch.randint(im_size[1]*args.canvas_size, (1,))), (1,2))[:,:im_size[0],:im_size[1]] for im in x]) for _ in range(args.canvas_samples)])
                    this_y = torch.cat([this_y for _ in range(args.canvas_samples)])

                if args.dsa and (not args.no_aug):
                    x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

                if args.distributed:
                    forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
                else:
                    forward_params = student_params[-1]

                x = student_net(x, flat_param=forward_params)
                ce_loss = criterion(x, this_y)

                grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]

                student_params.append(student_params[-1] - syn_lr * grad)

            ''' *Update synthetic dataset'''
            param_loss = torch.tensor(0.0).to(args.device)
            param_dist = torch.tensor(0.0).to(args.device)

            param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
            param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

            param_loss_list.append(param_loss)
            param_dist_list.append(param_dist)

            param_loss /= num_params
            param_dist /= num_params

            param_loss /= param_dist

            grand_loss = param_loss

            optimizer_img.zero_grad()
            optimizer_lr.zero_grad()

            grand_loss.backward()

            optimizer_img.step()
            optimizer_lr.step()

            wandb.log({"Grand_Loss": grand_loss.detach().cpu(),
                       "Random_Start_Epoch": rn_start_epoch})


            for _ in student_params:
                del _

            if it%10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))


    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--ipc', type=int, default=1, help='case1: equal image(s) per class in each segments')
    parser.add_argument('--ipcs', type=str, default=None, help='case2: unequal image(s) per class in each segments')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')

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

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--max_end_epoch', type=int, default=25, help='the final expert epoch we should learn')

    parser.add_argument('--intervals', type=str, default=None,help='fixed intervals to segment student trajectory')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

    parser.add_argument('--reparam_syn', action='store_true')
    parser.add_argument('--run_name', type=str, default=None, help="run_name")
    parser.add_argument('--pre_names', type=str, default=None, help="The names of previous subsets")
    parser.add_argument('--lrs_net', type=str, default=None, help="The lrs of previous subsets")
    parser.add_argument('--name', type=str, default=None, help="The name of this subset")

    args = parser.parse_args()

    main(args)

