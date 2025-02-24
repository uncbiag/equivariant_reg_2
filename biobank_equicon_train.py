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

input_shape = [1, 1, 64, 156, 224]


DO_HARD_AUGMENT = False



def make_batch(dataset, pairs_made):
    image = torch.cat([random.choice(dataset) for _ in range(GPUS * BATCH_SIZE)])
    image = image.cuda()
    image = image / torch.max(image)
    print(torch.max(image), torch.min(image))

    if DO_HARD_AUGMENT:
        return augment(image, pairs_made)
    else:
        return image

def augment(image_A, pairs_made):

    noise = torch.randn((image_A.shape[0], 3, 3))

    noise = noise - torch.permute(noise, (0, 2, 1))

    forward = torch.linalg.matrix_exp(noise * pairs_made / 10000)

    full = torch.zeros((image_A.shape[0], 3, 4))

    full[:, :3, :3] = forward

    grid_shape = list(image_A.shape)
    grid_shape[1] = 3
    forward_grid = F.affine_grid(full.cuda(), grid_shape)
    print(full)

    warped_A = F.grid_sample(image_A, forward_grid, padding_mode="border")

    return warped_A
def make_make_pair(dataset):

    pairs_made = [0]
    def make_pair():
        pairs_made[0] += 1
        return make_batch(dataset, pairs_made[0]), make_batch(dataset, pairs_made[0])
    return make_pair

if __name__ == "__main__":
    import equivariant_reg


    dataset = torch.load(
        "./results/preproecss_biobank/training_data.trch")
    footsteps.initialize()
    threestep_consistent_net = equivariant_reg.make_network_final(input_shape, dimension=3, diffusion=True)
    threestep_consistent_net.lmbda=10

    net_par = torch.nn.DataParallel(threestep_consistent_net).cuda()
    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.0001)

    net_par.train()

    BATCH_SIZE = 1
    GPUS = 4


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
