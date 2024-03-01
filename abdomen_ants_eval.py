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
parser.add_argument("--writeimages", action="store_true")
args = parser.parse_args()




def preprocess(image):
    try:
        image = itk.CastImageFilter[itk.Image[itk.SS, 3], itk.Image[itk.F, 3]].New()(image)
    except:
        pass
    max_ = np.max(np.array(image))
    image = itk.shift_scale_image_filter(image, shift=0.0, scale=1.0 / max_)
    # image = itk.clamp_image_filter(image, bounds=(0, 1))
    return image


input_shape = [1, 1, 175, 175, 175]

dices = []
flips = []

import glob

def get_validation_images():
    res = []
    for i in range(800,888):
        name = f"/playpen-raid2/Data/AbdomenCT-1K/HastingsProcessed/results/pad_color_fix-4/val/cropped_masks/Case_{i:05}_0000.nii.gz"
        try:
            itk.imread(name)
        except:
            continue
        res.append(f"{i:05}")
    return res

def get_image(n):
    name = f"/playpen-raid2/Data/AbdomenCT-1K/HastingsProcessed/results/pad_color_fix-4/val/cropped_imgs/Case_{n}_0000.nii.gz"
    return itk.imread(name)



def get_sub_seg(n):
    name = f"/playpen-raid2/Data/AbdomenCT-1K/HastingsProcessed/results/pad_color_fix-4/val/cropped_masks/Case_{n}_0000.nii.gz"
    mask = itk.imread(name)
    return mask

atlas_registered = get_validation_images()

def mean_dice_f(sA, sB):
    sA, sB = [itk.image_from_array(s.numpy()) for s in (sA, sB)]
    return utils.itk_mean_dice(sA, sB)

import ants

random.seed(1)
for _ in range(30):
    n_A, n_B = (random.choice(atlas_registered) for _ in range(2))
    image_A, image_B = (preprocess(get_image(n)) for n in (n_A, n_B))
    segmentation_A, segmentation_B = (get_sub_seg(n) for n in (n_A, n_B))

    # import pdb; pdb.set_trace()
    import time
    start = time.time()


    image_A, image_B  = [ants.from_numpy(itk.array_from_image(t)) for t in (image_A, image_B)]

    segmentation_A, segmentation_B = [ants.from_numpy(itk.array_from_image(t).astype(np.uintc)) for t in (segmentation_A, segmentation_B)]

    
    
    reg_res_initial = ants.registration(segmentation_A, segmentation_B, "Affine", outprefix=footsteps.output_dir + "fo1r")
    reg_res = ants.registration(image_A, image_B, "SyN", outprefix=footsteps.output_dir + "for", initial_transform = reg_res_initial['fwdtransforms'])
    warped_segmentation_B = ants.apply_transforms(fixed=segmentation_A, moving=segmentation_B, transformlist=reg_res['fwdtransforms'], interpolator="nearestNeighbor")


    end = time.time()

    mean_dice = mean_dice_f(segmentation_A, warped_segmentation_B)

    print("time", end - start)



    utils.log(_)
    utils.log(n_A, n_B)
    utils.log(mean_dice)

    dices.append(mean_dice)

utils.log("Mean DICE")
utils.log(np.mean(dices))
