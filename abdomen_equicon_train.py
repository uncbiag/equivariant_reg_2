import random
import os
import torch
import numpy as np
import torch.nn.functional as F

import footsteps

import icon_registration as icon
import icon_registration.networks as networks
import footsteps

import icon_registration as icon
import icon_registration.data
import icon_registration.networks as networks
from icon_registration.config import device

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils


import icon_registration as icon

input_shape = [1, 1, 175, 175, 175]



def make_batch(dataset):
    image = torch.cat([random.choice(dataset) for _ in range(GPUS * BATCH_SIZE)])
    image = image.cuda()
    image = image / torch.max(image)
    print(torch.max(image), torch.min(image))
    return image

def make_make_pair(dataset):
    def make_pair():
        return make_batch(dataset), make_batch(dataset)
    return make_pair

if __name__ == "__main__":
    import equivariant_reg

    footsteps.initialize()
    threestep_consistent_net = equivariant_reg.make_network_final(input_shape, dimension=3, diffusion=True)

    net_par = torch.nn.DataParallel(threestep_consistent_net).cuda()
    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.0001)

    net_par.train()

    BATCH_SIZE = 1
    GPUS = 4


    dataset = torch.load("/playpen-raid2/Data/AbdomenCT-1K/HastingsProcessed/results/pad_fix_traintensor/train_imgs_tensor.trch")
    icon.train_batchfunction(
        net_par,
        optimizer,
        make_make_pair(dataset),
        unwrapped_net=threestep_consistent_net,
        steps=1500
    )
    old_state = threestep_consistent_net.state_dict()
    threestep_consistent_net = equivariant_reg.make_network_final(input_shape, dimension=3, diffusion=False)
    threestep_consistent_net.load_state_dict(old_state)
    net_par = torch.nn.DataParallel(threestep_consistent_net).cuda()

    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.0001)

    net_par.train()
    icon.train_batchfunction(
        net_par,
        optimizer,
        make_make_pair(dataset),
        unwrapped_net=threestep_consistent_net,
        steps=50000,
    )
