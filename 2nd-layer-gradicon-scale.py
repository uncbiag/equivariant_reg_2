#!/usr/bin/env python
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


def make_im(input_shape):
    input_shape = np.array(input_shape)
    input_shape[0] = 1
    spacing = 1.0 / (input_shape[2::] - 1)
    _id = identity_map_multiN(input_shape, spacing)
    return _id


def pad_im(im, n):
    new_shape = np.array(im.shape)
    old_shape = np.array(im.shape)
    new_shape[2:] += 2 * n
    new_im = np.array(make_im(new_shape))
    if len(new_shape) == 4:

        def expand(t):
            return t[None, 2:, None, None]

    else:

        def expand(t):
            return t[None, 2:, None, None, None]

    new_im *= expand((new_shape - 1)) / expand((old_shape - 1))
    new_im -= n / expand((old_shape - 1))
    new_im = torch.tensor(new_im)
    return new_im


class AttentionRegistration(icon_registration.RegistrationModule):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.dim = 128

        self.padding = 9

    def crop(self, x):
        padding = self.padding
        return x[:, :, padding:-padding, padding:-padding]

    def featurize(self, values, recrop=True):
        padding = self.padding
        x = torch.nn.functional.pad(values, [padding, padding, padding, padding])
        x = self.net(x)
        x = 4 * x / (0.001 + torch.sqrt(torch.sum(x**2, dim=1, keepdims=True)))
        if recrop:
            x = self.crop(x)
        return x

    def torch_attention(self, ft_A, ft_B):
        ft_A = ft_A.reshape(
            -1,
            1,
            self.dim,
            (self.identity_map.shape[-1] + 2 * self.padding)
            * (self.identity_map.shape[-2] + 2 * self.padding),
        )
        ft_B = ft_B.reshape(
            -1, 1, self.dim, self.identity_map.shape[-1] * self.identity_map.shape[-2]
        )
        ft_A = ft_A.permute([0, 1, 3, 2]).contiguous()
        ft_B = ft_B.permute([0, 1, 3, 2]).contiguous()
        im = pad_im(self.identity_map, self.padding).to(ft_A.device)
        x = im.reshape(-1, 1, 2, ft_A.shape[2]).permute(0, 1, 3, 2)
        x = torch.cat([x, x], axis=-1)
        x = x.expand(10, -1, -1, -1)
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            output = torch.nn.functional.scaled_dot_product_attention(
                ft_B, ft_A, x, scale=1
            )
        output = output[:, :, :, 2:]
        output = output.permute(0, 1, 3, 2)
        output = output.reshape(
            -1, 2, self.identity_map.shape[2], self.identity_map.shape[3]
        )
        return output

    def forward(self, A, B):
        ft_A = self.featurize(A, recrop=False)
        ft_B = self.featurize(B)
        output = self.torch_attention(ft_A, ft_B)
        output = output - self.identity_map
        # output = self.blur_kernel(output)
        return output


ds1, ds2 = icon_registration.data.get_dataset_retina(
    include_boundary=False, scale=0.8, fixed_vertical_offset=200, split="test"
)
sample_batch = next(iter(ds2))[0]
plt.imshow(torchvision.utils.make_grid(sample_batch[:12], nrow=4)[0])
footsteps.plot("data sample")

image_A = sample_batch.cuda()

unet = alias_free_unet.NoDownsampleNet()
ar = AttentionRegistration(unet)
inner_net = icon.network_wrappers.DownsampleRegistration(
    icon.network_wrappers.DownsampleRegistration(icon.FunctionFromVectorField(ar), 2), 2
)
ts = icon.TwoStepRegistration(
    icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=2)),
    icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=2)),
)
ts = TwoStepLayerwiseRegularizer(
    DiffusionLayerwiseRegularizer(inner_net, 1.5),
    GradientICONLayerwiseRegularizer(ts, 1.5),
)

net = CollectLayerwiseRegularizer(ts, icon.LNCC(sigma=4))
net.assign_identity_map(sample_batch.shape)
net.cuda()


def show(tensor):
    plt.imshow(torchvision.utils.make_grid(tensor[:6], nrow=3)[0].cpu().detach())
    plt.xticks([])
    plt.yticks([])


image_A = next(iter(ds1))[0].to(device)
image_B = next(iter(ds2))[0].to(device)
net(image_A, image_B)
plt.subplot(2, 2, 1)
show(image_A)
plt.subplot(2, 2, 2)
show(image_B)
plt.subplot(2, 2, 3)
show(net.warped_image_A)
plt.contour(
    torchvision.utils.make_grid(net.phi_AB_vectorfield[:6], nrow=3)[0].cpu().detach()
)
plt.contour(
    torchvision.utils.make_grid(net.phi_AB_vectorfield[:6], nrow=3)[1].cpu().detach()
)
plt.subplot(2, 2, 4)
show(net.warped_image_A - image_B)
show(net.phi_AB_vectorfield[:, 1])
plt.colorbar()

footsteps.plot("before_registration")

net.train()
net.to(device)
optim = torch.optim.Adam(net.parameters(), lr=0.0003)
curves = icon.train_datasets(net, optim, ds1, ds2, epochs=45)
plt.plot(np.array(curves)[:, :3])

footsteps.plot("train curve")
plt.plot(np.array(curves)[250:, :3])

footsteps.plot("train curve")




image_A = next(iter(ds1))[0].to(device)
image_B = next(iter(ds2))[0].to(device)
net(image_A, image_B)
plt.subplot(2, 2, 1)
show(image_A)
plt.subplot(2, 2, 2)
show(image_B)
plt.subplot(2, 2, 3)
show(net.warped_image_A)
plt.contour(
    torchvision.utils.make_grid(net.phi_AB_vectorfield[:6], nrow=3)[0].cpu().detach()
)
plt.contour(
    torchvision.utils.make_grid(net.phi_AB_vectorfield[:6], nrow=3)[1].cpu().detach()
)
plt.subplot(2, 2, 4)
show(net.warped_image_A - image_B)

footsteps.plot("after_registration")


firststep_net = icon.losses.DiffusionRegularizedNet(
    inner_net, icon.LNCC(sigma=4), lmbda=1.5
)
firststep_net.assign_identity_map(sample_batch.shape)
firststep_net.cuda()



image_A = next(iter(ds1))[0].to(device)
image_B = next(iter(ds2))[0].to(device)
firststep_net(image_A, image_B)
plt.subplot(2, 2, 1)
show(image_A)
plt.subplot(2, 2, 2)
show(image_B)
plt.subplot(2, 2, 3)
show(firststep_net.warped_image_A)
plt.contour(
    torchvision.utils.make_grid(firststep_net.phi_AB_vectorfield[:6], nrow=3)[0]
    .cpu()
    .detach()
)
plt.contour(
    torchvision.utils.make_grid(firststep_net.phi_AB_vectorfield[:6], nrow=3)[1]
    .cpu()
    .detach()
)
plt.subplot(2, 2, 4)
show(firststep_net.warped_image_A - image_B)
footsteps.plot("just transformer")

features = unet(torch.nn.functional.avg_pool2d(image_A, 2).cuda())
for i in range(14):
    show(features[:, i:])
    footsteps.plot("features")
ds1, ds2 = icon_registration.data.get_dataset_retina(
    include_boundary=False, scale=0.8, fixed_vertical_offset=200
)
image_A = next(iter(ds1))[0].to(device)
image_B = next(iter(ds2))[0].to(device)
net(image_A, image_B)
plt.subplot(2, 2, 1)
show(image_A)
plt.subplot(2, 2, 2)
show(image_B)
plt.subplot(2, 2, 3)
show(net.warped_image_A)
plt.contour(
    torchvision.utils.make_grid(net.phi_AB_vectorfield[:6], nrow=3)[0].cpu().detach()
)
plt.contour(
    torchvision.utils.make_grid(net.phi_AB_vectorfield[:6], nrow=3)[1].cpu().detach()
)
plt.subplot(2, 2, 4)
show(net.warped_image_A - image_B)

footsteps.plot("after_registration")
