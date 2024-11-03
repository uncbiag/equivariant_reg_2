import icon_registration as icon
import numpy as np
import torch
import torch.linalg

# here be dragons. 
# specifically this forces torch to import things in the right order in a multithreaded context. 
# (probably)
z = torch.linalg.inv(torch.tensor([[1., 0], [0, 1]]).cuda())

class RandomMatrix(icon.RegistrationModule):
    def forward(self, a, b):
        if len(a.shape) == 4:
            noise = torch.randn(a.shape[0], 2, 2) * 13
            noise = noise - noise.permute([0, 2, 1])
            noise = torch.linalg.matrix_exp(noise)
            noise = torch.cat([noise, torch.zeros(a.shape[0], 2, 1)], axis=2).to(a.device)
            x = noise
            x = torch.cat(
                [
                    x,
                    torch.Tensor([[[0, 0, 1]]]).to(x.device).expand(x.shape[0], -1, -1),
                ],
                1,
            )
            x = torch.matmul(
                torch.Tensor([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]).to(x.device), x
            )
            x = torch.matmul(
                x,
                torch.Tensor([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]]).to(x.device),
            )
            return x
        elif len(a.shape) == 5:
            noise = torch.randn(a.shape[0], 3, 3) * 13
            noise = noise - noise.permute([0, 2, 1])
            noise = torch.linalg.matrix_exp(noise)
            noise = torch.cat([noise, torch.zeros(a.shape[0], 3, 1)], axis=2).to(a.device)
            x = noise
            x = torch.cat(
                [
                    x,
                    torch.Tensor([[[0, 0, 0, 1]]]).to(x.device).expand(x.shape[0], -1, -1),
                ],
                1,
            )
            x = torch.matmul(
                torch.Tensor([[1, 0, 0, 0.5], [0, 1, 0, 0.5], [0, 0, 1, 0.5], [0, 0, 0, 1]]).to(x.device), x
            )
            x = torch.matmul(
                x,
                torch.Tensor([[1, 0, 0, -0.5], [0, 1, 0, -0.5], [0, 0, 1, -0.5], [0, 0, 0, 1]]).to(x.device),
            )
            return x

class FunctionsFromMatrix(icon.RegistrationModule):
    """
    wrap an inner neural network `net` that returns an N x N+1 matrix representing
    an affine transform, into a RegistrationModule that returns a function that
    transforms a tensor of coordinates.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image_A, image_B):
        matrix_phi = self.net(image_A, image_B).detach().clone()
        matrix_phi = np.array(matrix_phi.cpu().detach())
        matrix_phi = torch.tensor(matrix_phi).to(image_A.device)

        def transform(tensor_of_coordinates):
            shape = list(tensor_of_coordinates.shape)
            shape[1] = 1
            coordinates_homogeneous = torch.cat(
                [tensor_of_coordinates, torch.ones(shape, device=tensor_of_coordinates.device)], axis=1
            )
            return icon.network_wrappers.multiply_matrix_vectorfield(matrix_phi, coordinates_homogeneous)[:, :-1]

        inv = torch.linalg.inv(matrix_phi.detach().clone())

        def invtransform(tensor_of_coordinates):
            shape = list(tensor_of_coordinates.shape)
            shape[1] = 1
            coordinates_homogeneous = torch.cat(
                [tensor_of_coordinates, torch.ones(shape, device=tensor_of_coordinates.device)], axis=1
            )
            return icon.network_wrappers.multiply_matrix_vectorfield(inv, 
                    coordinates_homogeneous)[:, :-1]

        return transform, invtransform

class PostStep(icon.RegistrationModule):

    def __init__(self, netPhi, netPsi):
        super().__init__()
        self.netPhi = netPhi
        self.netPsi = netPsi

    def forward(self, image_A, image_B):
        
        # Tag for shortcutting hack. Must be set at the beginning of 
        # forward because it is not preserved by .to(config.device)
        self.identity_map.isIdentity = True
            
        phi, invphi = self.netPhi(image_A, image_B)
        psi = self.netPsi(
            image_A,
            self.as_function(image_B)(invphi(self.identity_map)),
        )
        return lambda tensor_of_coordinates: psi(phi(tensor_of_coordinates))
