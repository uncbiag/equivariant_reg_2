from fixed_point_carl import show
from icon_registration.config import device
import fixed_point_carl as fpc
import footsteps
import icon_registration as icon
import icon_registration.carl as carl
import icon_registration.data
import icon_registration.networks as networks
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import torchvision.utils

input_shape = [1, 1, 175, 175, 175]
def make_net(dimension, input_shape):
    unet = fpc.Equivariantize(fpc.SomeDownsampleNoDilationNet(dimension=dimension))
    ar = fpc.AttentionFeaturizer(unet, dimension=dimension)
    ts = ar

    ts = icon.network_wrappers.DownsampleNet(ts, dimension)
    for _ in range(3):
         ts = icon.TwoStepRegistration(
             fpc.Blur(ts, 21),
             #ts,
             icon.network_wrappers.DownsampleNet(carl.RotationFunctionFromVectorField(networks.tallUNet2(dimension=dimension)), dimension)
         )
    for _ in range(2):
         ts = icon.TwoStepRegistration(
             fpc.Blur(ts, 11),
             #ts,
             icon.network_wrappers.DownsampleNet(carl.RotationFunctionFromVectorField(networks.tallUNet2(dimension=dimension)), dimension)
         )
    ts = icon.network_wrappers.DownsampleNet(ts, dimension)
    #ts = icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=dimension))

    for _ in range(1):
         ts = icon.TwoStepRegistration(
             fpc.Blur(ts, 8),
             #ts,
             carl.RotationFunctionFromVectorField(networks.tallUNet2(dimension=dimension))
         )
    #net = icon.losses.GradientICONSparse(ts, icon.LNCC(sigma=4), lmbda=1.5)
    net = icon.losses.DiffusionRegularizedNet(ts, icon.LNCC(sigma=4), lmbda=10)
    net.assign_identity_map(input_shape)
    #net = carl.augmentify(net)
    #net.assign_identity_map(input_shape)
    net.train()
    return net

def make_batch(dataset):
    batch = [random.choice(dataset).cuda() for _ in range(GPUS * BATCH_SIZE)]
    batch = [image / torch.max(image) for image in batch]
    image = torch.cat(batch)
    #print(torch.max(image), torch.min(image))

    return image

def make_make_pair(dataset):

    def make_pair():
        return make_batch(dataset), make_batch(dataset)
    return make_pair



if __name__ == "__main__":

    dataset = torch.load("/playpen-raid2/Data/AbdomenCT-1K/HastingsProcessed/results/stretched_traintensor/train_imgs_tensor.trch"
    )
    #dataset = torch.load(
    #        "smol_dataset.trch"
    #)
    footsteps.initialize()

    net = make_net(3, input_shape)
    #net.regis_net.load_state_dict(torch.load("results/abdomen_no_dilation/network_weights_900"))
    #net = torch.compile(net)
    net = net.cuda()

    #net_par = torch.nn.DataParallel(net).cuda()
    net_par = net
    optimizer = torch.optim.Adam(net_par.parameters(), lr=0.0001)

    net_par.train()

    BATCH_SIZE = 4
    GPUS = 1

    #image_A, image_B = make_make_pair(dataset)()

    #import torch._dynamo as dynamo

    #explanation = dynamo.explain(net_par)(image_A, image_B)

    #print(explanation)
    



    icon.train_batchfunction(
        net_par,
        optimizer,
        make_make_pair(dataset),
        unwrapped_net=net,
        steps=45000
    )

