import matplotlib.pyplot as plt
import torchvision.utils

def show(tensor):
    plt.imshow(torchvision.utils.make_grid(tensor[:6], nrow=3)[0].cpu().detach())
    plt.xticks([])
    plt.yticks([])

def visualize(image_A, image_B, net):
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
