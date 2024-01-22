import icon_registration
import icon_registration as icon
import icon_registration.data
import icon_registration.networks as networks
from icon_registration.config import device
import numpy as np
import torch
import torchvision.utils
import matplotlib.pyplot as plt
import alias_free_unet
import importlib
import icon_registration.network_wrappers as network_wrappers
from icon_registration.losses import ICONLoss, flips
from icon_registration.mermaidlite import identity_map_multiN
from layerwise_regularizer import TwoStepLayerwiseRegularizer, GradientICONLayerwiseRegularizer, DiffusionLayerwiseRegularizer, CollectLayerwiseRegularizer


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
    new_im *= expand((new_shape - 1 )) / expand((old_shape - 1))
    new_im -= n / expand((old_shape - 1))
    new_im = torch.tensor(new_im)
    return new_im

class AttentionRegistration(icon_registration.RegistrationModule):
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
                self.identity_map.shape[-1]
                * self.identity_map.shape[-2]
                * self.identity_map.shape[-3],
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
                self.identity_map.shape[-1]
                * self.identity_map.shape[-2]
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
                self.identity_map.shape[2],
                self.identity_map.shape[3],
                self.identity_map.shape[4],
            )
        elif self.dimension == 2:
            output = output.reshape(
                -1,
                2,
                self.identity_map.shape[2],
                self.identity_map.shape[3],
            )
        return output
        
    def forward(self, A, B):
        ft_A = self.featurize(A, recrop=False)   
        ft_B = self.featurize(B)
        output = self.torch_attention(ft_A, ft_B)    
        output = output  - self.identity_map  
        return output

def make_network(input_shape, dimension):
    unet = alias_free_unet.NoDownsampleNet(dimension = dimension)
    ar = AttentionRegistration(unet, dimension=dimension)
    inner_net = icon.network_wrappers.DownsampleRegistration(
      icon.network_wrappers.DownsampleRegistration(
        icon.FunctionFromVectorField(ar), dimension), dimension)
    ts = icon.TwoStepRegistration(
        icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=dimension)),
        icon.FunctionFromVectorField(icon.networks.tallUNet2(dimension=dimension)))
    ts = TwoStepLayerwiseRegularizer(
        DiffusionLayerwiseRegularizer(inner_net, 1.5),
        GradientICONLayerwiseRegularizer(ts, 1.5))
        
    net = CollectLayerwiseRegularizer(ts, icon.LNCC(sigma=4))
    net.assign_identity_map(input_shape)
    net.cuda()
    return net
