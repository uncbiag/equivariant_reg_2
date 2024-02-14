import argparse
import os
import random

import footsteps
footsteps.initialize(output_root="evaluation_results/")
import icon_registration as icon
import icon_registration.itk_wrapper as itk_wrapper
import itk
import numpy as np
import torch

import utils

parser = argparse.ArgumentParser()
parser.add_argument("weights_path" )
parser.add_argument("--finetune", action="store_true")
parser.add_argument("--writeimages", action="store_true")
args = parser.parse_args()

weights_path = args.weights_path


def preprocess(image):
    image = itk.CastImageFilter[type(image), itk.Image[itk.F, 3]].New()(image)
    print(type(image))
    max_ = np.max(np.array(image))
    image = itk.shift_scale_image_filter(image, shift=0.0, scale=1.0 / max_)
    # image = itk.clamp_image_filter(image, bounds=(0, 1))
    return image


input_shape = [1, 1, 175, 175, 175]
import equivariant_reg

net = equivariant_reg.make_network([1, 1, 175, 175, 175], 3)

#multiscale_constr_model.multiscale_affine_model
#
#qq = torch.nn.Module()
#qq.module = net
utils.log(net.regis_net.load_state_dict(torch.load(weights_path), strict=True))
net.eval()

dices = []
flips = []
ICON_errors=[]

import glob

def get_validation_images():
    res = []
    for i in range(800,900):
        name = f"/playpen-raid2/Data/AbdomenCT-1K/HastingsProcessed/results/processed_imgs/val/cropped_masks/Case_{i:05}_0000.nii.gz"
        try:
            itk.imread(name)
        except:
            continue
        res.append(f"{i:05}")
    return res

def get_image(n):
    name = f"/playpen-raid2/Data/AbdomenCT-1K/HastingsProcessed/results/processed_imgs/val/cropped_imgs/Case_{n}_0000.nii.gz"
    return itk.imread(name)



def get_sub_seg(n):
    name = f"/playpen-raid2/Data/AbdomenCT-1K/HastingsProcessed/results/processed_imgs/val/cropped_masks/Case_{n}_0000.nii.gz"
    mask = itk.imread(name)
    return mask

atlas_registered = get_validation_images()



random.seed(1)
for _ in range(30):
    n_A, n_B = (random.choice(atlas_registered) for _ in range(2))
    image_A, image_B = (preprocess(get_image(n)) for n in (n_A, n_B))

    # import pdb; pdb.set_trace()
    import time
    start = time.time()
    phi_AB, phi_BA, loss = itk_wrapper.register_pair(
        net,
        image_A,
        image_B,
        finetune_steps=(50 if args.finetune == True else None),
        return_artifacts=True,
    )
    end = time.time()

    print("time", end - start)

    segmentation_A, segmentation_B = (get_sub_seg(n) for n in (n_A, n_B))

    interpolator = itk.NearestNeighborInterpolateImageFunction.New(segmentation_A)

    warped_segmentation_A = itk.resample_image_filter(
        segmentation_A,
        transform=phi_AB,
        interpolator=interpolator,
        use_reference_image=True,
        reference_image=segmentation_B,
    )
    mean_dice = utils.itk_mean_dice(segmentation_B, warped_segmentation_A)
    if args.writeimages:
        casedir = footsteps.output_dir + str(_) + "/" 
        os.mkdir(casedir)

        itk.imwrite(image_A, casedir + "imageA.nii.gz")
        itk.imwrite(image_B, casedir + "imageB.nii.gz")
        itk.imwrite(segmentation_A, casedir + "segmentation_A.nii.gz")
        itk.imwrite(segmentation_B, casedir + "segmentation_B.nii.gz")
        itk.imwrite(warped_segmentation_A, casedir+ "warpedseg.nii.gz")
        itk.transformwrite([phi_AB], casedir + "trans.hdf5")

    utils.log(_)
    utils.log(n_A, n_B)
    utils.log(mean_dice)

    dices.append(mean_dice)
    flips.append(loss.flips)
    scale=150
    zz = (net.phi_AB(net.phi_BA(net.identity_map)) - net.identity_map) * scale
    icon_error = torch.mean(torch.sqrt(torch.sum(zz**2, axis=1))).item()
    ICON_errors.append(icon_error)
    utils.log("ICON_error", icon_error)
utils.log("mean ICON error", np.mean(ICON_errors))

utils.log("Mean DICE")
utils.log(np.mean(dices))
utils.log("Mean flips")
utils.log(np.mean(flips))
utils.log("flips / prod(imnput_shape", np.mean(flips) / np.prod(input_shape))

utils.log("percent J", 100 * np.mean(flips) / np.prod(input_shape))
