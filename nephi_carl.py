#!/usr/bin/env python
from icon_registration import config
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
import icon_registration
import icon_registration as icon
import icon_registration.data
import icon_registration.networks as networks
from icon_registration.config import device
import numpy as np
import torch
import torchvision.utils
import matplotlib.pyplot as plt
import no_downsample_net
import importlib
import icon_registration.network_wrappers as network_wrappers
from icon_registration.losses import ICONLoss, flips
from icon_registration.mermaidlite import identity_map_multiN
from layerwise_regularizer import TwoStepLayerwiseRegularizer, GradientICONLayerwiseRegularizer, DiffusionLayerwiseRegularizer, CollectLayerwiseRegularizer


from layerwise_regularizer import *
from equivariant_reg import AttentionRegistration
from equivariant_reg import pad_im, make_im
import custom_retina_dataset


ds1, ds2 = custom_retina_dataset.get_dataset_retina(
)
sample_batch = next(iter(ds2))[0]
plt.imshow(torchvision.utils.make_grid(sample_batch[:12], nrow=4)[0])
footsteps.plot("data sample")

image_A = sample_batch.cuda()


import equivariant_reg

class SparseAttentionRegistration(icon_registration.RegistrationModule):
    def __init__(self, net, dimension=2):
        super().__init__()
        self.net = net
        self.dim = 128
        self.dimension = dimension
        
        self.padding = 9

    def crop(self, x):
        padding = self.padding
        if self.dimension == 3:
            return x[:, :, padding:-padding, padding:-padding, padding:-padding]
        elif self.dimension == 2:
            return x[:, :, padding:-padding, padding:-padding]
    
    def featurize(self, values, recrop=True):       
        padding = self.padding
        if self.dimension == 3:
            x = torch.nn.functional.pad(
                values, [padding, padding, padding, padding, padding, padding]
            )
        elif self.dimension == 2:
            x = torch.nn.functional.pad(
                values, [padding, padding, padding, padding]
            )
        x = self.net(x)        
        x = 4 * x / (.001 + torch.sqrt(torch.sum(x**2, dim=1, keepdims=True)))        
        if recrop:
            x = self.crop(x)
        return x
    def torch_attention(self, ft_A, ft_B):
        ft_B_shape = ft_B.shape
        if self.dimension == 3:
            ft_A = ft_A.reshape(
                -1,
                1,
                self.dim,
               ( self.identity_map.shape[-1] + 2 * self.padding)
                * (self.identity_map.shape[-2] + 2 * self.padding)
                * (self.identity_map.shape[-3] + 2 * self.padding))
            ft_B = ft_B.reshape(
                -1,
                1,
                self.dim,
                ft_B_shape[-1]
                * ft_B_shape[-2]
                * ft_B_shape[-3],
            )
        elif self.dimension == 2:
            ft_A = ft_A.reshape(
                -1,
                1,
                self.dim,
               ( self.identity_map.shape[-1] + 2 * self.padding)
                * (self.identity_map.shape[-2] + 2 * self.padding)
                )
            ft_B = ft_B.reshape(
                -1,
                1,
                self.dim,
                ft_B_shape[-1]
                * ft_B_shape[-2]
                ,
            )
        ft_A = ft_A.permute([0, 1, 3, 2]).contiguous()
        ft_B = ft_B.permute([0, 1, 3, 2]).contiguous()
        im = pad_im(self.identity_map, self.padding).to(ft_A.device)
        x = im.reshape(-1, 1, self.dimension, ft_A.shape[2]).permute(0, 1, 3, 2)
        x = torch.cat([x, x], axis=-1)
        x = torch.cat([x, x], axis=-1)
        x = x[:, :, :, :4]
        x = x.expand(ft_A.shape[0], -1, -1, -1).contiguous()
        #print(ft_A.stride(), ft_B.stride(), x.stride())
        #print(ft_A.shape, ft_B.shape, x.shape)

        with torch.backends.cuda.sdp_kernel(enable_math=False):
            output = torch.nn.functional.scaled_dot_product_attention(ft_B, ft_A, x, scale=1)
        output = output[:, :, :, :self.dimension]
        output = output.permute(0, 1, 3, 2)
        if self.dimension == 3:
            output = output.reshape(
                -1,
                3,
                ft_B_shape[2],
                ft_B_shape[3],
                ft_B_shape[4],
            )
        elif self.dimension == 2:
            output = output.reshape(
                -1,
                2,
                ft_B_shape[2],
                ft_B_shape[3],
            )
        return output
        

        return output

    def forward(self, image_A, image_B):
        ft_A = self.featurize(image_A, recrop=False)   
        ft_B = self.featurize(image_B)


        def transform(coordinates):
            coordinate_features = self.as_function(ft_B)(coordinates)
            output = self.torch_attention(ft_A, coordinate_features)    

            return output

        return transform

class GradientICONSparse(network_wrappers.RegistrationModule):
    def __init__(self, network, similarity, lmbda):

        super().__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity

    def forward(self, image_A, image_B):

        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)
        
        sparse_coords = torch.rand(image_A.shape[0], self.identity_map.shape[1], 5424, 1).cuda()

        phi_AB_vectorfield = self.phi_AB(sparse_coords)
        phi_BA_vectorfield = self.phi_BA(sparse_coords)

        # tag images during warping so that the similarity measure
        # can use information about whether a sample is interpolated
        # or extrapolated

        if getattr(self.similarity, "isInterpolated", False):
            # tag images during warping so that the similarity measure
            # can use information about whether a sample is interpolated
            # or extrapolated
            inbounds_tag = torch.zeros([image_A.shape[0]] + [1] + list(image_A.shape[2:]), device=image_A.device)
            if len(self.input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(self.input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0
        else:
            inbounds_tag = None

        warped_image_A = self.as_function(image_A)(phi_AB_vectorfield)
        unwarped_image_B = self.as_function(image_B)(sparse_coords)

        warped_image_B = self.as_function(image_B)(phi_BA_vectorfield)
        unwarped_image_A = self.as_function(image_A)(sparse_coords)

        similarity_loss = self.similarity(
            warped_image_A, unwarped_image_B
        ) + self.similarity(warped_image_B, unwarped_image_A)

        Iepsilon = sparse_coords


        # compute squared Frobenius of Jacobian of icon error

        direction_losses = []

        approximate_Iepsilon = self.phi_AB(phi_BA_vectorfield)

        inverse_consistency_error = Iepsilon - approximate_Iepsilon

        delta = 0.001

        if len(self.identity_map.shape) == 4:
            dx = torch.Tensor([[[[delta]], [[0.0]]]]).to(config.device)
            dy = torch.Tensor([[[[0.0]], [[delta]]]]).to(config.device)
            direction_vectors = (dx, dy)

        elif len(self.identity_map.shape) == 5:
            dx = torch.Tensor([[[[[delta]]], [[[0.0]]], [[[0.0]]]]]).to(config.device)
            dy = torch.Tensor([[[[[0.0]]], [[[delta]]], [[[0.0]]]]]).to(config.device)
            dz = torch.Tensor([[[[0.0]]], [[[0.0]]], [[[delta]]]]).to(config.device)
            direction_vectors = (dx, dy, dz)
        elif len(self.identity_map.shape) == 3:
            dx = torch.Tensor([[[delta]]]).to(config.device)
            direction_vectors = (dx,)

        for d in direction_vectors:
            approximate_Iepsilon_d = self.phi_AB(self.phi_BA(Iepsilon + d))
            inverse_consistency_error_d = Iepsilon + d - approximate_Iepsilon_d
            grad_d_icon_error = (
                inverse_consistency_error - inverse_consistency_error_d
            ) / delta
            direction_losses.append(torch.mean(grad_d_icon_error**2))

        inverse_consistency_loss = sum(direction_losses)

        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss

        transform_magnitude = torch.mean(
            (sparse_coords - phi_AB_vectorfield) ** 2
        )
        print( ICONLoss(
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            flips(phi_BA_vectorfield),
        ))
        return ICONLoss(
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            flips(phi_BA_vectorfield),
        )
    def prepare_for_viz(self, image_A, image_B):
        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        self.phi_BA = self.regis_net(image_B, image_A)
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map)

        self.warped_image_A = self.as_function(image_A)(self.phi_AB_vectorfield)
        self.warped_image_B = self.as_function(image_B)(self.phi_BA_vectorfield)

class DiffusionRegularizedSparse(network_wrappers.RegistrationModule):
    def __init__(self, network, similarity, lmbda):

        super().__init__()

        self.regis_net = network
        self.lmbda = lmbda
        self.similarity = similarity

    def forward(self, image_A, image_B):


        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True

        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)
        
        sparse_coords = torch.rand(image_A.shape[0], self.identity_map.shape[1], 5424, 1).cuda()

        phi_AB_vectorfield = self.phi_AB(sparse_coords)
        phi_BA_vectorfield = self.phi_BA(sparse_coords)

        # tag images during warping so that the similarity measure
        # can use information about whether a sample is interpolated
        # or extrapolated

        if getattr(self.similarity, "isInterpolated", False):
            # tag images during warping so that the similarity measure
            # can use information about whether a sample is interpolated
            # or extrapolated
            inbounds_tag = torch.zeros([image_A.shape[0]] + [1] + list(image_A.shape[2:]), device=image_A.device)
            if len(self.input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(self.input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0
        else:
            inbounds_tag = None

        warped_image_A = self.as_function(image_A)(phi_AB_vectorfield)
        unwarped_image_B = self.as_function(image_B)(sparse_coords)

        warped_image_B = self.as_function(image_B)(phi_BA_vectorfield)
        unwarped_image_A = self.as_function(image_A)(sparse_coords)

        similarity_loss = self.similarity(
            warped_image_A, unwarped_image_B
        ) + self.similarity(warped_image_B, unwarped_image_A)

        Iepsilon = sparse_coords


        # compute squared Frobenius of Jacobian of icon error

        direction_losses = []

        approximate_Iepsilon = phi_BA_vectorfield

        inverse_consistency_error = Iepsilon - approximate_Iepsilon

        delta = 0.001

        if len(self.identity_map.shape) == 4:
            dx = torch.Tensor([[[[delta]], [[0.0]]]]).to(config.device)
            dy = torch.Tensor([[[[0.0]], [[delta]]]]).to(config.device)
            direction_vectors = (dx, dy)

        elif len(self.identity_map.shape) == 5:
            dx = torch.Tensor([[[[[delta]]], [[[0.0]]], [[[0.0]]]]]).to(config.device)
            dy = torch.Tensor([[[[[0.0]]], [[[delta]]], [[[0.0]]]]]).to(config.device)
            dz = torch.Tensor([[[[0.0]]], [[[0.0]]], [[[delta]]]]).to(config.device)
            direction_vectors = (dx, dy, dz)
        elif len(self.identity_map.shape) == 3:
            dx = torch.Tensor([[[delta]]]).to(config.device)
            direction_vectors = (dx,)

        for d in direction_vectors:
            approximate_Iepsilon_d = self.phi_BA(Iepsilon + d)
            inverse_consistency_error_d = Iepsilon + d - approximate_Iepsilon_d
            grad_d_icon_error = (
                inverse_consistency_error - inverse_consistency_error_d
            ) / delta
            direction_losses.append(torch.mean(grad_d_icon_error**2))

        inverse_consistency_loss = sum(direction_losses)

        all_loss = self.lmbda * inverse_consistency_loss + similarity_loss

        transform_magnitude = torch.mean(
            (sparse_coords - phi_AB_vectorfield) ** 2
        )
        print( ICONLoss(
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            flips(phi_BA_vectorfield),
        ))
        return ICONLoss(
            all_loss,
            inverse_consistency_loss,
            similarity_loss,
            transform_magnitude,
            flips(phi_BA_vectorfield),
        )
    def prepare_for_viz(self, image_A, image_B):
        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        self.phi_BA = self.regis_net(image_B, image_A)
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map)

        self.warped_image_A = self.as_function(image_A)(self.phi_AB_vectorfield)
        self.warped_image_B = self.as_function(image_B)(self.phi_BA_vectorfield)

def make_just_transformer_network(input_shape, dimension, diffusion=False):
    ar = SparseAttentionRegistration(no_downsample_net.NoDownsampleNet())

    ts = icon.DownsampleRegistration(icon.DownsampleRegistration(ar, 2), 2)

    if diffusion:
        net = DiffusionRegularizedSparse (ts, icon.NCC(), 1.5)
    else:
        net = GradientICONSparse(ts, icon.NCC(), 1.5)
        
    net.assign_identity_map(input_shape)
    net.cuda()
    return net



net = make_just_transformer_network(sample_batch.shape, 2, diffusion=True)


ds1, ds2 = custom_retina_dataset.get_dataset_retina(split="train")

title = "a title"

image_A = next(iter(ds1))[0].to(device)
image_B = next(iter(ds2))[0].to(device)
net.prepare_for_viz(image_A, image_B)

visualize(image_A, image_B, net)

footsteps.plot(title + "before_registration")

net.train()
net.to(device)
optim = torch.optim.Adam(net.parameters(), lr=0.0003)
curves = icon.train_datasets(net, optim, ds1, ds2, epochs=1)

plt.plot(np.array(curves)[:, :3])
footsteps.plot(title + "train curves")
state_dict = net.state_dict()

net = make_just_transformer_network(sample_batch.shape, 2, diffusion=False)
net.load_state_dict(state_dict)
net.train()
net.to(device)
optim = torch.optim.Adam(net.parameters(), lr=0.0003)
curves = icon.train_datasets(net, optim, ds1, ds2, epochs=6)


plt.plot(np.array(curves)[:, :3])
footsteps.plot(title + "train curves")

net.prepare_for_viz(image_A, image_B)
visualize(image_A, image_B, net)
footsteps.plot(title + "after_registration")


