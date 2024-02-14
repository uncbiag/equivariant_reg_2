import torch

def DICE(image_A, image_B):
    image_A = image_A > .5
    image_B = image_B > .5

    return 2 * torch.sum(image_A * image_B) / (torch.sum(image_A) + torch.sum(image_B))
