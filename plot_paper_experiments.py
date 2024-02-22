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

input_dir = "results/paper_expr_repeat/trial"

        

# we padd the non shifted retina dataset always on one side to make the shape the same. ✓

# The models we need are predict displacement, transformer, hybrid ✓

# We want a good train test split for the retina dataset ✓

# We want landmarks in the retina dataset. Can do later, start with DICE. Can do later, start with LNCC.

from dice import DICE

def make_hybrid_network(input_shape, dimension, diffusion=False):
    unet = no_downsample_net.NoDownsampleNet(dimension = dimension)
    ar = AttentionRegistration(unet, dimension=dimension)
    inner_net = icon.network_wrappers.DownsampleRegistration(
      icon.network_wrappers.DownsampleRegistration(
        icon.FunctionFromVectorField(ar), dimension), dimension)
    ts = icon.TwoStepRegistration(
        icon.DownsampleRegistration(icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=dimension)), dimension=dimension),
        icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=dimension)))
    ts = icon.network_wrappers.TwoStepRegistration(inner_net, ts)

    if diffusion:
        net = icon.losses.DiffusionRegularizedNet(ts, icon.LNCC(4), 1.5)
    else:
        net = icon.losses.GradientICON(ts, icon.LNCC(4), 1.5)
        
    net.assign_identity_map(input_shape)
    net.cuda()
    return net

def make_just_transformer_network(input_shape, dimension, diffusion=False):
    unet = no_downsample_net.NoDownsampleNet(dimension = dimension)
    ar = AttentionRegistration(unet, dimension=dimension)
    ts = icon.network_wrappers.DownsampleRegistration(
      icon.network_wrappers.DownsampleRegistration(
        icon.FunctionFromVectorField(ar), dimension), dimension)

    if diffusion:
        net = icon.losses.DiffusionRegularizedNet(ts, icon.LNCC(4), 1.5)
    else:
        net = icon.losses.GradientICON(ts, icon.LNCC(4), 1.5)
        
    net.assign_identity_map(input_shape)
    net.cuda()
    return net

def make_just_displacement_network(input_shape, dimension, diffusion=False):
    inner_net = icon.network_wrappers.DownsampleRegistration(
      icon.network_wrappers.DownsampleRegistration(
        icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension)), dimension), dimension)
    ts = icon.TwoStepRegistration(
        icon.DownsampleRegistration(icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=dimension)), dimension=dimension),
        icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=dimension)))
    ts = icon.network_wrappers.TwoStepRegistration(inner_net, ts)

    if diffusion:
        net = icon.losses.DiffusionRegularizedNet(ts, icon.LNCC(4), 1.5)
    else:
        net = icon.losses.GradientICON(ts, icon.LNCC(4), 1.5)
        
    net.assign_identity_map(input_shape)
    net.cuda()
    return net


results = {}

for experiment_name, datasets in experiments.items():

    for network_fn in [make_hybrid_network, make_just_transformer_network, make_just_displacement_network]:
        net_name = network_fn.__name__
        title = experiment_name + " " + net_name + " "
        results[title] = []

        for training_run in range(1, 20):

            ds1, ds2 = custom_retina_dataset.get_dataset_retina(split="train", **dataset_kwargs[datasets["train"]])



            image_A = next(iter(ds1))[0].to(device)
            image_B = next(iter(ds2))[0].to(device)

            net = network_fn(sample_batch.shape, 2, diffusion=False)
            state_dict = torch.load(input_dir + "-" + str(training_run) + "/" + title + "network_weights")
            net.load_state_dict(state_dict)
            net.to(device)

            #visualize(image_A, image_B, net)
            #footsteps.plot(title + "after_registration")


            ds1, ds2 = custom_retina_dataset.get_dataset_retina(split="test", **dataset_kwargs[datasets["test"]])

            ds1, ds2 = iter(ds1), iter(ds2)
            dices = []

            for i in range(1):
                image_A = next(ds1)[0].to(device)
                image_B = next(ds2)[0].to(device)


                visualize(image_A, image_B, net)

                dice = DICE(net.warped_image_A > .2, image_B > .2).item()
                dices.append(dice)

                footsteps.plot(title + "test")
            
            results[title].append(np.mean(dices))


def adjust_title(title):
    title = title.replace("_network", "")
    title = title.replace("make_", "")
    title = title.replace("just_", "")
    title = title.replace(" ", "\n")
    title = title.replace("_", " ")
    return title

torch.save(results, footsteps.output_dir + "dice_results.trch")

print(results)
        
        
plt.bar(list(map(adjust_title, results.keys())), results.values())        
plt.xticks(rotation=90)
plt.tight_layout()
footsteps.plot(name="final dices")
