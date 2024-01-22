
from icon_registration.losses import ICONLoss, flips
import icon_registration.network_wrappers as network_wrappers
import torch
from icon_registration.config import device

class GradientICONLayerwiseRegularizer(network_wrappers.RegistrationModule):
    def __init__(self, network, lmbda):
        super().__init__()
        self.regis_net = network
        self.lmbda = lmbda

    def compute_gradient_icon_loss(self, phi_AB, phi_BA):
        Iepsilon = (
            self.identity_map
            + torch.randn(*self.identity_map.shape).to(self.identity_map.device)
            * 1
            / self.identity_map.shape[-1]
        )
        direction_losses = []
        approximate_Iepsilon = phi_AB(phi_BA(Iepsilon))
        inverse_consistency_error = Iepsilon - approximate_Iepsilon
        delta = 0.001
        if len(self.identity_map.shape) == 4:
            dx = torch.Tensor([[[[delta]], [[0.0]]]]).to(self.identity_map.device)
            dy = torch.Tensor([[[[0.0]], [[delta]]]]).to(self.identity_map.device)
            direction_vectors = (dx, dy)
        elif len(self.identity_map.shape) == 5:
            dx = torch.Tensor([[[[[delta]]], [[[0.0]]], [[[0.0]]]]]).to(
                self.identity_map.device
            )
            dy = torch.Tensor([[[[[0.0]]], [[[delta]]], [[[0.0]]]]]).to(
                self.identity_map.device
            )
            dz = torch.Tensor([[[[0.0]]], [[[0.0]]], [[[delta]]]]).to(
                self.identity_map.device
            )
            direction_vectors = (dx, dy, dz)
        elif len(self.identity_map.shape) == 3:
            dx = torch.Tensor([[[delta]]]).to(self.identity_map.device)
            direction_vectors = (dx,)
        for d in direction_vectors:
            approximate_Iepsilon_d = phi_AB(phi_BA(Iepsilon + d))
            inverse_consistency_error_d = Iepsilon + d - approximate_Iepsilon_d
            grad_d_icon_error = (
                inverse_consistency_error - inverse_consistency_error_d
            ) / delta
            direction_losses.append(torch.mean(grad_d_icon_error**2))
        inverse_consistency_loss = sum(direction_losses)
        return inverse_consistency_loss

    def forward(self, image_A, image_B):
        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]
        self.identity_map.isIdentity = True
        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_BA = self.regis_net(image_B, image_A)
        inverse_consistency_loss = self.compute_gradient_icon_loss(
            self.phi_AB, self.phi_BA
        )
        return self.phi_AB, self.lmbda * inverse_consistency_loss + torch.mean(
            10000 * (self.phi_AB(self.identity_map) - self.identity_map) ** 2
        )


class DiffusionLayerwiseRegularizer(network_wrappers.RegistrationModule):
    def __init__(self, network, lmbda):
        super().__init__()
        self.regis_net = network
        self.lmbda = lmbda

    def compute_diffusion_loss(self, phi_AB_vectorfield):
        phi_AB_vectorfield = self.identity_map - phi_AB_vectorfield
        if len(self.identity_map.shape) == 3:
            bending_energy = torch.mean(
                (-phi_AB_vectorfield[:, :, 1:] + phi_AB_vectorfield[:, :, 1:-1]) ** 2
            )
        elif len(self.identity_map.shape) == 4:
            bending_energy = torch.mean(
                (-phi_AB_vectorfield[:, :, 1:] + phi_AB_vectorfield[:, :, :-1]) ** 2
            ) + torch.mean(
                (-phi_AB_vectorfield[:, :, :, 1:] + phi_AB_vectorfield[:, :, :, :-1])
                ** 2
            )
        elif len(self.identity_map.shape) == 5:
            bending_energy = (
                torch.mean(
                    (-phi_AB_vectorfield[:, :, 1:] + phi_AB_vectorfield[:, :, :-1]) ** 2
                )
                + torch.mean(
                    (
                        -phi_AB_vectorfield[:, :, :, 1:]
                        + phi_AB_vectorfield[:, :, :, :-1]
                    )
                    ** 2
                )
                + torch.mean(
                    (
                        -phi_AB_vectorfield[:, :, :, :, 1:]
                        + phi_AB_vectorfield[:, :, :, :, :-1]
                    )
                    ** 2
                )
            )
        return bending_energy * self.identity_map.shape[2] ** 2

    def forward(self, image_A, image_B):
        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]
        self.identity_map.isIdentity = True
        self.phi_AB = self.regis_net(image_A, image_B)
        diffusion_loss = self.compute_diffusion_loss(self.phi_AB(self.identity_map))
        return self.phi_AB, self.lmbda * diffusion_loss


class TwoStepLayerwiseRegularizer(network_wrappers.RegistrationModule):
    def __init__(self, phi, psi):
        super().__init__()
        self.phi = phi
        self.psi = psi

    def forward(self, image_A, image_B):
        phi_AB, loss1 = self.phi(image_A, image_B)
        a_circ_phi_AB = self.as_function(image_A)(phi_AB(self.identity_map))
        psi_AB, loss2 = self.psi(a_circ_phi_AB, image_B)
        return (lambda coords: phi_AB(psi_AB(coords))), loss1 + loss2


class CollectLayerwiseRegularizer(network_wrappers.RegistrationModule):
    def __init__(self, network, similarity):
        super().__init__()
        self.regis_net = network
        self.similarity = similarity

    def compute_similarity_measure(self, phi_AB_vectorfield, image_A, image_B):
        if getattr(self.similarity, "isInterpolated", False):
            inbounds_tag = torch.zeros(
                [image_A.shape[0]] + [1] + list(image_A.shape[2:]),
                device=image_A.device,
            )
            if len(self.input_shape) - 2 == 3:
                inbounds_tag[:, :, 1:-1, 1:-1, 1:-1] = 1.0
            elif len(self.input_shape) - 2 == 2:
                inbounds_tag[:, :, 1:-1, 1:-1] = 1.0
            else:
                inbounds_tag[:, :, 1:-1] = 1.0
        else:
            inbounds_tag = None
        self.warped_image_A = self.as_function(
            torch.cat([image_A, inbounds_tag], axis=1)
            if inbounds_tag is not None
            else image_A
        )(phi_AB_vectorfield)

        similarity_loss = self.similarity(self.warped_image_A, image_B)
        return similarity_loss

    def forward(self, image_A, image_B) -> ICONLoss:
        assert self.identity_map.shape[2:] == image_A.shape[2:]
        assert self.identity_map.shape[2:] == image_B.shape[2:]
        self.identity_map.isIdentity = True
        self.phi_AB, regularity_loss = self.regis_net(image_A, image_B)
        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)

        similarity_loss = 2 * self.compute_similarity_measure(
            self.phi_AB_vectorfield, image_A, image_B
        )
        all_loss = regularity_loss + similarity_loss
        transform_magnitude = torch.mean(
            (self.identity_map - self.phi_AB_vectorfield) ** 2
        )
        return ICONLoss(
            all_loss,
            regularity_loss,
            similarity_loss,
            transform_magnitude,
            flips(self.phi_AB_vectorfield),
        )

    def prepare_for_viz(self, image_A, image_B):
        self.phi_AB = self.regis_net(image_A, image_B)
        self.phi_AB_vectorfield = self.phi_AB(self.identity_map)
        self.phi_BA = self.regis_net(image_B, image_A)
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map)
        self.warped_image_A = self.as_function(image_A)(self.phi_AB_vectorfield)
        self.warped_image_B = self.as_function(image_B)(self.phi_BA_vectorfield)
