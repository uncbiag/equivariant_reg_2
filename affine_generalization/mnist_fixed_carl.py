import footsteps
import icon_registration as icon
import icon_registration.data
import icon_registration.networks as networks
from icon_registration.config import device
import icon_registration.carl as carl


import numpy as np
import torch
import torchvision.utils
import matplotlib.pyplot as plt
import fixed_point_carl as fpc
from fixed_point_carl import show


ds, _ = icon_registration.data.get_dataset_mnist(split="train", number=2)

sample_batch = next(iter(ds))[0]
plt.imshow(torchvision.utils.make_grid(sample_batch[:12], nrow=4)[0])


unet = fpc.Equivariantize(fpc.SomeDownsampleNet(dimension=2))
ar = fpc.AttentionFeaturizer(unet, dimension=2)
ts = ar

#ts = icon.network_wrappers.DownsampleNet(carl.RotationFunctionFromVectorField(networks.tallUNet2(dimension=2)), 2)
for _ in range(3):
     ts = icon.TwoStepRegistration(
	 fpc.Blur(ts, 12),
	 #ts,
	 icon.network_wrappers.DownsampleNet(carl.RotationFunctionFromVectorField(networks.tallUNet2(dimension=2)), 2)
     )


for _ in range(3):
     ts = icon.TwoStepRegistration(
	 fpc.Blur(ts, 8),
	 #ts,
	 carl.RotationFunctionFromVectorField(networks.tallUNet2(dimension=2))
     )
net = icon.losses.GradientICONSparse(ts, icon.LNCC(sigma=4), lmbda=3)
net.assign_identity_map(sample_batch.shape)
net = carl.augmentify(net)
net.cuda()
net.train()
optim = torch.optim.Adam(net.parameters(), lr=0.0005)


curves = icon.train_datasets(net, optim, ds, ds, epochs=10)
plt.close()
plt.plot(np.array(curves)[:, :3])


footsteps.plot("curves")



image_A = next(iter(ds))[0].to(device)
image_B = next(iter(ds))[0].to(device)
net(image_A, image_B)


plt.subplot(2, 2, 1)
show(image_A)
plt.subplot(2, 2, 2)
show(image_B)
plt.subplot(2, 2, 3)
show(net.warped_image_A)
plt.contour(torchvision.utils.make_grid(net.phi_AB_vectorfield[:6], nrow=3)[0].cpu().detach())
plt.contour(torchvision.utils.make_grid(net.phi_AB_vectorfield[:6], nrow=3)[1].cpu().detach())
plt.subplot(2, 2, 4)
show(net.warped_image_A - image_B)
plt.tight_layout()
footsteps.plot("result")
