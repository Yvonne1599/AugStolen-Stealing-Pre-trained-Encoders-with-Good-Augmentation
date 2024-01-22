import torch
import argparse
from network import ResNET
from loadmodel import *
from loaddata import *
from metric import *


parser = argparse.ArgumentParser(description='Check the performance of the pre-trained encoder')
parser.add_argument('--gpu', default=0, type=int, help='Gpu index.')
parser.add_argument('--task', default='cifar10')
parser.add_argument('--ch', default='clean')
parser.add_argument('--arch', default='res50')
parser.add_argument('--ssl', default ='simclr')
parser.add_argument('--path', default='./log/total/')
args = parser.parse_args()



if torch.cuda.is_available():
    device = torch.device("cuda:%d"%args.gpu)
else:   
    device = torch.device("cpu")


def main():
    
    # This part checks the performance of different versions of encoders, with the use of downstream classifier for some classification tasks.

    if args.ch == 'clean':
        model = load_victim(args.ssl, device)
    elif args.ch == 'supervised':
        model = ResNET(args.arch)
    elif args.ch == 'surrogate':
        model = load_surrogate(args.arch, args.path, device)
        
    with torch.cuda.device(args.gpu):
        # Test the accuracy in downstream classification tasks.
        # Here, the tasks indicate different datasets.
        DA = classify(args.task, model, device)
        print('Downstream accuracy is ', DA)


if __name__ == "__main__":
    main()
    