# from loaddata import load_pridata
import torch
import torch.nn as nn
import argparse
import time
import torch.optim as optim
import os
from network import *
from loadmodel import *
from loaddata import *


torch.backends.cudnn.benchmark = True
torch.set_printoptions(profile="full")


parser = argparse.ArgumentParser(
    description="AugStolen for encoder stealing attacks")
parser.add_argument('--gpu', default=0, type=int, help='GPU ID.')
parser.add_argument('--epochs', type=int, default=100,
                    help='epochs (default: 100)')
parser.add_argument('--aug', type=bool, default=True,
                    help='augmented data (default: True)')
parser.add_argument('--lam', type=float, default=1,
                    help='lambda for the augmented data in loss function')
parser.add_argument('--model-dir', default='./log/',
                    help='address for saving images')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--rst', type=bool, default=True)
parser.add_argument('--dist', type=str, default='cosine')
parser.add_argument('--victim', default='simclr')
parser.add_argument('--log-dir', default='./log/')
parser.add_argument('--shadow-path', default='./shadow/')
parser.add_argument('--log-name', default='log.txt')
parser.add_argument('--path', default='./log/total/')
args = parser.parse_args()


PATH_SAVE = args.model_dir
if not os.path.exists(PATH_SAVE):
    os.makedirs(PATH_SAVE)

PATH_LOG = args.log_dir
if not os.path.exists(PATH_LOG):
    os.makedirs(PATH_LOG)


lr_shadow = 1e-2

if args.dist == 'cosine':
    dist = nn.CosineSimilarity(dim=1, eps=1e-6)
elif args.dist == 'l2':
    dist = nn.PairwiseDistance(p=2, eps=1e-6)
elif args.dist == 'l1':
    dist = nn.PairwiseDistance(p=1, eps=1e-6)

# cos = nn.CosineSimilarity(dim=1, eps=1e-6)

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda:%d" % args.gpu)
else:
    device = torch.device("cpu")


if __name__ == '__main__':

    torch.cuda.synchronize()
    start = time.time()

    if args.rst == True:
        print("Stealing based on a raw model.")
        model_shadow = ResNET('res50')
    else:
        model_shadow = load_surrogate("res50", args.shadow_path, device)
        print("Stealing based on a pretrained model.")

    model_victim = load_victim(args.victim, device)

    for param in model_victim.parameters():
        param.requires_grad = False
    for param in model_shadow.parameters():
        param.requires_grad = True

    optimizer_sd = optim.SGD(model_shadow.parameters(),
                             lr=lr_shadow, momentum=0.9)

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer_sd, gamma=0.96)

    train_loader, test_loader = load_data('cifar10-2k', 16)
    aug_train_loader, aug_test_loader = load_data('cifar10-2k', 16, True)

    # This part is for encoder stealing.

    for epoch in range(args.epochs):

        for i, (data1, data2) in enumerate(zip(train_loader, aug_train_loader)):

            img_pretrain, _ = data1
            img_aug, _ = data2

            if use_cuda:
                img_pretrain = img_pretrain.to(device)
                img_aug = img_aug.to(device)

                model_victim = model_victim.to(device)
                model_shadow = model_shadow.to(device)

            optimizer_sd.zero_grad()

            model_victim.eval()
            model_shadow.train()
            h_1_vc = torch.squeeze(model_victim(img_pretrain))
            h_1_sd = torch.squeeze(model_shadow(img_pretrain))
            h_1_sd_aug = torch.squeeze(model_shadow(img_aug))

            if args.dist == 'cosine':
                jacobian = torch.autograd.functional.jacobian(
                    -dist, h_1_sd.unsqueeze(0)).squeeze()

                h_1_sd_jabocian = torch.squeeze(
                    model_shadow(img_pretrain + jacobian))

                Ls = -torch.mean(dist(h_1_vc, h_1_sd))
                Ls_aug = -torch.mean(dist(h_1_vc, h_1_sd_aug))
                Ls_Jacobian = -torch.mean(dist(h_1_vc, h_1_sd_jabocian))
            elif args.dist == 'l2':
                jacobian = torch.autograd.functional.jacobian(
                    dist, h_1_sd.unsqueeze(0)).squeeze()

                h_1_sd_jabocian = torch.squeeze(
                    model_shadow(img_pretrain + jacobian))

                Ls = torch.mean(dist(h_1_vc, h_1_sd))
                Ls_aug = torch.mean(dist(h_1_vc, h_1_sd_aug))
                Ls_Jacobian = torch.mean(dist(h_1_vc, h_1_sd_jabocian))
            elif args.dist == 'l1':
                jacobian = torch.autograd.functional.jacobian(
                    dist, h_1_sd.unsqueeze(0)).squeeze()

                h_1_sd_jabocian = torch.squeeze(
                    model_shadow(img_pretrain + jacobian))

                Ls = torch.mean(dist(h_1_vc, h_1_sd))
                Ls_aug = torch.mean(dist(h_1_vc, h_1_sd_aug))
                Ls_Jacobian = torch.mean(dist(h_1_vc, h_1_sd_jabocian))

            total_loss = Ls + args.lam * (Ls_aug + Ls_Jacobian)

            total_loss.backward(retain_graph=True)
            optimizer_sd.step()

        print("Training epoch: %d, Part 1 Loss: %.4f Part 2 Loss: %.4f Total Loss: %.4f" % (
            epoch+1, Ls.item(), Ls_aug.item(), total_loss.item()))
        lr_scheduler.step()

        # SAVE

        if (epoch+1) % 10 == 0:
            ID = str(epoch+1+args.start)

            if args.rst == True:
                checkpoint_name = 'shadow_'+ID + "_aug_" + args.dist + '.pth.tar'
            else:
                checkpoint_name = 'shadow_'+ID + "_aug" + '_continue_' + args.dist + '.pth.tar'

            filename = os.path.join(PATH_SAVE+checkpoint_name)
            torch.save(model_shadow.state_dict(), filename)
