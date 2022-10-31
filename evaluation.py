import os
import argparse
import numpy as np
import torch
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import copy

def main(args):
    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(
        args.dataset, args.data_path, args.batch_real, args.subset, args=args)

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

    # modi:
    image_path = os.path.join(args.syn_image_path, args.dataset, args.run_name)
    image_folder = os.listdir(image_path)
    print(image_folder)
    img_l = []
    lab_l = []
    for f in image_folder:
        print(f)
        max_epoch = max([eval(_.split('.')[0].split('_')[1]) for _ in os.listdir(image_path+'/'+f) if 'labels' in _ and 'best' not in _])
        img = torch.load(os.path.join(image_path,f,'images_{}.pt'.format(max_epoch)))
        lab = torch.load(os.path.join(image_path,f,'labels_{}.pt'.format(max_epoch)))
        print(max_epoch)
        img_l.append(img)
        lab_l.append(lab)
        break
    image_syn = torch.cat(img_l, dim=0)
    label_syn = torch.cat(lab_l, dim=0)

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    for model_eval in model_eval_pool:
        if args.dsa:
            print('DSA augmentation strategy: \n', args.dsa_strategy)
            print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
        else:
            print('DC augmentation parameters: \n', args.dc_aug_param)

        accs_test = []
        accs_train = []
        for it_eval in range(args.num_eval):
            net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)  # get a random model
            args.lr_net = 0.033
            eval_labs = label_syn
            with torch.no_grad():
                image_save = image_syn
            image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(
                eval_labs.detach())  # avoid any unaware modification

            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args,
                                                     texture=args.texture)
            accs_test.append(acc_test)
            accs_train.append(acc_train)
        accs_test = np.array(accs_test)
        accs_train = np.array(accs_train)
        acc_test_mean = np.mean(accs_test)
        acc_test_std = np.std(accs_test)
        print(acc_test_mean)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

    # modi:
    parser.add_argument('--syn_image_path', type=str, default='logged_files', help='buffer path')
    parser.add_argument('--run_name', type=str, default='fearless-valley-7', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=3, help='max expert epochs number the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')

    # modi: 分段参数
    parser.add_argument('--overlap', type=int, default=1, help='the overlap num of params between segment student params')
    parser.add_argument('--segments', type=int, default=5, help='Number of student trajectory segments')
    parser.add_argument('--interval', type=int, default=None, help='using fixed interval length to segment student trajectory')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')

    args = parser.parse_args()

    main(args)

