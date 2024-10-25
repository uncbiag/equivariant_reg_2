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
    threestep_consistent_net = equivariant_reg.make_network_final_final(input_shape, dimension=3, diffusion=True)

    net_par = torch.nn.DataParallel(threestep_consistent_net).cuda()
    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.0001)

    net_par.train()

    BATCH_SIZE = 3
    GPUS = 4


    dataset = torch.load("/work/users/t/g/tgreer/train_imgs_tensor.trch")

    old_state = torch.load("/users/t/g/tgreer/network_weights_48900")
    threestep_consistent_net = equivariant_reg.make_network_final_final(input_shape, dimension=3, diffusion=False)
    threestep_consistent_net.regis_net.load_state_dict(old_state)
    threestep_consistent_net.regis_net = icon.TwoStepRegistration(threestep_consistent_net.regis_net,
            icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=3)))
    fourstep_consistent_net = threestep_consistent_net
    fourstep_consistent_net.assign_identity_map(input_shape)
    fourstep_consistent_net.train()
    net_par = torch.nn.DataParallel(fourstep_consistent_net).cuda()

    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.0001)

    net_par.train()
    icon.train_batchfunction(
        net_par,
        optimizer,
        dataloader.make_batch,
        unwrapped_net=threestep_consistent_net,
        steps=50000,
    )
