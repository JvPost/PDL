import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions.normal as dist

import time

from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

trainset = datasets.MNIST(root='./data',download=True, train=True, transform=transform)
testset = datasets.MNIST(root='./data', download=True, transform=transform)

# Prepare the training set
train_features = torch.cat([data[0].view((1,28*28)) for data in trainset])
train_labels = torch.Tensor([data[1] for data in trainset]).type(torch.long)




def compute_marginalized_predictions(layer, features, num_samples):
    weight_samples = layer.get_weight_samples(layer.mu_W, layer.s_W, num_samples)
    pre_preds = layer.linear(weight_samples, features)
    preds = nn.Softmax(dim=2)(pre_preds)
    return torch.mean(preds, 1)