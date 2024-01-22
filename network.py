import torch
import torch.nn as nn
import torchvision
# Downstream Classifier. Used for classfication on datasets with 10 different classes.


class C10(torch.nn.Module):
    def __init__(self):
        super(C10, self).__init__()
        self.backdone = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            # Change it when you need use it on other datasets with different classes.
            nn.Linear(256, 10),
        )

    def forward(self, x):
        y = self.backdone(x)
        return y

# Several ResNet-based encoders, based on ResNet in torch.nn.Module.
# The size of the embedding is 2048.


class myResNet18(nn.Module):
    def __init__(self):
        super(myResNet18, self).__init__()
        self.backbone = torchvision.models.resnet18(
            pretrained=True, num_classes=1000)
        self.backbone.fc = nn.Identity()
        self.backbone.maxpool = nn.Identity()
        self.fc1 = nn.Linear(2048, 2048)

    def forward(self, x):
        h = self.backbone(x)
        h = h.view(h.size(0), -1)
        y = self.fc1(h)
        return y


class myResNet34(nn.Module):
    def __init__(self):
        super(myResNet34, self).__init__()
        self.backbone = torchvision.models.resnet34(
            pretrained=True, num_classes=1000)
        self.backbone.fc = nn.Identity()
        self.backbone.maxpool = nn.Identity()
        self.fc1 = nn.Linear(2048, 2048)

    def forward(self, x):
        h = self.backbone(x)
        h = h.view(h.size(0), -1)
        y = self.fc1(h)
        return y


class myResNet50(nn.Module):
    def __init__(self):
        super(myResNet50, self).__init__()
        self.backbone = torchvision.models.resnet50(
            pretrained=True, num_classes=1000)
        self.backbone.fc = nn.Identity()
        self.backbone.maxpool = nn.Identity()
        self.fc1 = nn.Linear(2048, 2048)

    def forward(self, x):
        h = self.backbone(x)
        h = h.view(h.size(0), -1)
        y = self.fc1(h)
        return y


class myResNet101(nn.Module):
    def __init__(self):
        super(myResNet101, self).__init__()
        self.backbone = torchvision.models.resnet101(
            pretrained=True, num_classes=1000)
        self.backbone.fc = nn.Identity()
        self.backbone.maxpool = nn.Identity()
        self.fc1 = nn.Linear(2048, 2048)

    def forward(self, x):
        h = self.backbone(x)
        h = h.view(h.size(0), -1)
        y = self.fc1(h)
        return


class myResNet152(nn.Module):
    def __init__(self):
        super(myResNet101, self).__init__()
        self.backbone = torchvision.models.resnet152(
            pretrained=True, num_classes=1000)
        self.backbone.fc = nn.Identity()
        self.backbone.maxpool = nn.Identity()
        self.fc1 = nn.Linear(2048, 2048)

    def forward(self, x):
        h = self.backbone(x)
        h = h.view(h.size(0), -1)
        y = self.fc1(h)
        return y


def ResNET(arch):

    # Select different encoders based on ResNet.

    if arch == 'res18':
        model = myResNet18()
    elif arch == 'res34':
        model = myResNet34()
    elif arch == 'res50':
        model = myResNet50()
    elif arch == 'res101':
        model = myResNet101()
    elif arch == 'res152':
        model = myResNet152()

    return model
