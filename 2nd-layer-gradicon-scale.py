#!/usr/bin/env python
from visualize_2d import visualize
import footsteps
import math
from icon_registration.mermaidlite import identity_map_multiN
from icon_registration.losses import ICONLoss, flips
import icon_registration.network_wrappers as network_wrappers
import importlib
import no_downsample_net
import matplotlib.pyplot as plt
import torchvision.utils
import torch
import numpy as np
from icon_registration.config import device
import icon_registration.networks as networks
import icon_registration.data
import icon_registration as icon
import icon_registration
import os

from layerwise_regularizer import *


ds1, ds2 = icon_registration.data.get_dataset_retina(
    include_boundary=False, scale=0.8, fixed_vertical_offset=200
)
sample_batch = next(iter(ds2))[0]
plt.imshow(torchvision.utils.make_grid(sample_batch[:12], nrow=4)[0])
footsteps.plot("data sample")

image_A = sample_batch.cuda()


import equivariant_reg

net = equivariant_reg.make_network(sample_batch.shape, 2)




image_A = next(iter(ds1))[0].to(device)
image_B = next(iter(ds2))[0].to(device)

visualize(image_A, image_B, net)

footsteps.plot("before_registration")

net.train()
net.to(device)
optim = torch.optim.Adam(net.parameters(), lr=0.0003)
for _ in range(10):
    curves = icon.train_datasets(net, optim, ds1, ds2, epochs=4)


    visualize(image_A, image_B, net)
    footsteps.plot("after_registration")


    firststep_net = icon.losses.DiffusionRegularizedNet(
        net.regis_net.phi.regis_net, icon.LNCC(sigma=4), lmbda=1.5
    )
    firststep_net.assign_identity_map(sample_batch.shape)
    firststep_net.cuda()

    visualize(image_A, image_B, firststep_net)
    footsteps.plot("just transformer")

ds1, ds2 = icon_registration.data.get_dataset_retina(
    include_boundary=False, scale=0.8, fixed_vertical_offset=200, split="test"
)
image_A = next(iter(ds1))[0].to(device)
image_B = next(iter(ds2))[0].to(device)

visualize(image_A, image_B, net)
footsteps.plot("after_registration_test")

features = unet(torch.nn.functional.avg_pool2d(image_A, 2).cuda())
for i in range(14):
    show(features[:, i:])
    footsteps.plot("features")
