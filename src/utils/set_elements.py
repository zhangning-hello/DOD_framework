import torch
from torchvision import transforms
import torch.nn as nn

input_size_match = {
    'cifar100': [3, 32, 32],
    'cifar10': [3, 32, 32],
}


transforms_match = {

    'cifar100': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'cifar10': transforms.Compose([
        transforms.ToTensor(),
        ]),


}
