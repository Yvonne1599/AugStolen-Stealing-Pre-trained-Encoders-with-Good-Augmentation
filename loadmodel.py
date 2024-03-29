# This code is for the consine method
import torch
import torch.nn as nn
import torchvision
from resnet_wider import resnet50x1
from torchvision import models
from network import *

# It seems that different SSL learning methods should use the same model with architecture different.

# Load victim model trained by three different SSL methods.
# In our experiments, we here only focus on the function "load_victim".


def load_victim(encoder, device):

    if encoder == 'moco':

        model = torchvision.models.resnet50()

        for name, param in model.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                param.requires_grad = False

        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()

        encoder_path = "./victims/moco_official.pth.tar"

        checkpoint = torch.load(encoder_path, map_location=device)

        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            del state_dict[k]

        model.load_state_dict(state_dict, strict=False)
        model.fc = nn.Identity()
        model.maxpool = nn.Identity()

    elif encoder == 'simclr':

        model = resnet50x1()
        model = model.to(device)
        encoder_path = './victims/simclr_official.pth'
        checkpoint = torch.load(encoder_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.fc = nn.Identity()
        model.maxpool = nn.Identity()

    elif encoder == 'byol':

        class ResNet(torch.nn.Module):
            def __init__(self, net_name, pretrained=False, use_fc=False):
                super().__init__()
                base_model = models.__dict__[net_name](pretrained=pretrained)
                self.encoder = torch.nn.Sequential(
                    *list(base_model.children())[:-1])

                self.use_fc = use_fc
                if self.use_fc:
                    self.fc = torch.nn.Linear(2048, 512)

            def forward(self, x):
                x = self.encoder(x)
                x = torch.flatten(x, 1)
                if self.use_fc:
                    x = self.fc(x)
                return x

        model = ResNet('resnet50', pretrained=False, use_fc=False)

        # load encoder
        encoder_path = './victims/byol_official.pth.tar'
        checkpoint = torch.load(encoder_path, map_location=device)[
            'online_backbone']
        state_dict = {}
        length = len(model.encoder.state_dict())
        for name, param in zip(model.encoder.state_dict(), list(checkpoint.values())[:length]):
            state_dict[name] = param
        model.encoder.load_state_dict(state_dict, strict=True)

    return model


def load_surrogate(arch, encoder_path, device):
    if arch == 'res18':
        backbone = myResNet18()
        backbone.load_state_dict(torch.load(encoder_path, map_location=device))
    elif arch == 'res34':
        backbone = myResNet34()
        backbone.load_state_dict(torch.load(encoder_path, map_location=device))
    elif arch == 'res50':
        backbone = myResNet50()
        backbone.load_state_dict(torch.load(encoder_path, map_location=device))
    elif arch == 'res101':
        backbone = myResNet101()
        backbone.load_state_dict(torch.load(encoder_path, map_location=device))
    elif arch == 'res152':
        backbone = myResNet152()
        backbone.load_state_dict(torch.load(encoder_path, map_location=device))

    return backbone
