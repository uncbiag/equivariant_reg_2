#!/usr/bin/env python
from visualize_2d import visualize
import footsteps
import math
from icon_registration.mermaidlite import identity_map_multiN
from icon_registration.losses import ICONLoss, flips
import icon_registration.network_wrappers as network_wrappers
import importlib
import alias_free_unet
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
from equivariant_reg import AttentionRegistration
import custom_retina_dataset


ds1, ds2 = custom_retina_dataset.get_dataset_retina(
    scale=0.8, fixed_vertical_offset=200
)
sample_batch = next(iter(ds2))[0]
plt.imshow(torchvision.utils.make_grid(sample_batch[:12], nrow=4)[0])
footsteps.plot("data sample")

image_A = sample_batch.cuda()


import equivariant_reg


# The facts that we want to show:
# The equivariant transformer allows performance to not change when we add translation to the image
# The equivariant transformer allows performance to be acceptable when we add scale to the image.

# Performance is not acceptable with just predicting displacement.
# Predicting displacement works fine when both images are shifted together.

# So, the dataset we need is retina not shifted, retina shifted, retina shifted and scaled.

dataset_kwargs = {
        "not shifted" : { "fixed_vertical_offset": 200, "always_offset_bottom": True},
        "shifted" : {"fixed_vertical_offset": 200},
        "shifted_scaled": {"fixed_vertical_offset": 200, "scale": .8}
        }

experiments = {
        "baseline" : {"train":"not shifted", "test":"not shifted"},
        "tolerate_shift" : {"train":"shifted", "test":"shifted"},
        "generalize_shift" : {"train":"not shifted", "test":"shifted"},
        "hard_mode": {"train": "shifted_scaled", "test": "shifted_scaled"},
        "hard_mode_generalize": {"train": "not shifted", "test": "shifted_scaled"}
        }

        

# we padd the non shifted retina dataset always on one side to make the shape the same. ✓

# The models we need are predict displacement, transformer, hybrid ✓

# We want a good train test split for the retina dataset ✓

# We want landmarks in the retina dataset. Can do later, start with DICE. Can do later, start with LNCC.

def make_hybrid_network(input_shape, dimension):
    unet = alias_free_unet.NoDownsampleNet(dimension = dimension)
    ar = AttentionRegistration(unet, dimension=dimension)
    inner_net = icon.network_wrappers.DownsampleRegistration(
      icon.network_wrappers.DownsampleRegistration(
        icon.FunctionFromVectorField(ar), dimension), dimension)
    ts = icon.TwoStepRegistration(
        icon.DownsampleRegistration(icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=dimension)), dimension=dimension),
        icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=dimension)))
    ts = TwoStepLayerwiseRegularizer(
        DiffusionLayerwiseRegularizer(inner_net, .02),
        GradientICONLayerwiseRegularizer(ts, 1.5))
        
    net = CollectLayerwiseRegularizer(ts, icon.LNCC(sigma=4))
    net.assign_identity_map(input_shape)
    net.cuda()
    return net

def make_just_transformer_network(input_shape, dimension):
    unet = alias_free_unet.NoDownsampleNet(dimension = dimension)
    ar = AttentionRegistration(unet, dimension=dimension)
    inner_net = icon.network_wrappers.DownsampleRegistration(
      icon.network_wrappers.DownsampleRegistration(
        icon.FunctionFromVectorField(ar), dimension), dimension)
    ts = DiffusionLayerwiseRegularizer(inner_net, .02)
        
    net = CollectLayerwiseRegularizer(ts, icon.LNCC(sigma=4))
    net.assign_identity_map(input_shape)
    net.cuda()
    return net

def make_just_displacement_network(input_shape, dimension):
    inner_net = icon.network_wrappers.DownsampleRegistration(
      icon.network_wrappers.DownsampleRegistration(
        icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension)), dimension), dimension)
    ts = icon.TwoStepRegistration(
        icon.DownsampleRegistration(icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=dimension)), dimension=dimension),
        icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=dimension)))
    ts = TwoStepLayerwiseRegularizer(
        DiffusionLayerwiseRegularizer(inner_net, .02),
        GradientICONLayerwiseRegularizer(ts, 1.5))
        
    net = CollectLayerwiseRegularizer(ts, icon.LNCC(sigma=4))
    net.assign_identity_map(input_shape)
    net.cuda()
    return net



loss_curves = {}
for experiment_name, datasets in experiments.items():

    for network_fn in [make_hybrid_network, make_just_transformer_network, make_just_displacement_network]:


        net = network_fn(sample_batch.shape, 2)

        net_name = network_fn.__name__

        ds1, ds2 = custom_retina_dataset.get_dataset_retina(split="train", **dataset_kwargs[datasets["train"]])

        title = experiment_name + " " + net_name + " "


        image_A = next(iter(ds1))[0].to(device)
        image_B = next(iter(ds2))[0].to(device)

        #plt.imshow(torchvision.utils.make_grid(image_A[:12], nrow=4)[0].cpu())
        #footsteps.plot(title + "data sample A")
        #plt.imshow(torchvision.utils.make_grid(image_B[:12], nrow=4)[0].cpu())
        #footsteps.plot(title + "data sample B")

        #visualize(image_A, image_B, net)

        #footsteps.plot(title + "before_registration")

        net.train()
        net.to(device)
        optim = torch.optim.Adam(net.parameters(), lr=0.0003)
        curves = icon.train_datasets(net, optim, ds1, ds2, epochs=45)

        plt.plot(np.array(curves)[:, :3])
        footsteps.plot(title + "train curves")
        loss_curves[title] = curves

        visualize(image_A, image_B, net)
        footsteps.plot(title + "after_registration")


        ds1, ds2 = custom_retina_dataset.get_dataset_retina(split="test", **dataset_kwargs[datasets["test"]])

        image_A = next(iter(ds1))[0].to(device)
        image_B = next(iter(ds2))[0].to(device)

        #plt.imshow(torchvision.utils.make_grid(image_A[:12], nrow=4)[0].cpu())
        #footsteps.plot(title + "test data sample A")
        #plt.imshow(torchvision.utils.make_grid(image_B[:12], nrow=4)[0].cpu())
        #footsteps.plot(title + "test data sample B")

        visualize(image_A, image_B, net)

        footsteps.plot(title + "test")
        torch.save(net.state_dict(), footsteps.output_dir + title + "network_weights")

torch.save(loss_curves, footsteps.output_dir + "loss_curves")
