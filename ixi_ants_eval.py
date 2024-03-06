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

dices = []

def preprocess(image):
    image = itk.CastImageFilter[type(image), itk.Image[itk.F, 3]].New()(image)
    print(type(image))
    max_ = np.max(np.array(image))
    image = itk.shift_scale_image_filter(image, shift=0.0, scale=1.0 / max_)
    # image = itk.clamp_image_filter(image, bounds=(0, 1))
    return image

flips = []
ICON_errors=[]

import glob

def get_validation_images():
    res = []
    for i in range(428, 527):
        name = glob.glob( f"/playpen-raid1/tgreer/equivariant_reg_2/keymorph_comparison/keymorph/data/processed_IXI/T1/IXI{i:03}*")
        try:
            name[0]
        except:
            print(i, " failed")
            continue
        
        res.append(name[0])
    return res

def get_image(n):

    import subprocess
    #subprocess.run(["vshow", "-y", "-max", name])
    image = itk.imread(n)
    
    mask_name = n.replace("/T1/", "/T1_mask/")[:-4] + "_mask.nii.gz"

    mask = itk.imread(mask_name)

    mask = itk.CastImageFilter[type(mask), type(image)].New()(mask)

    mf = itk.MultiplyImageFilter[type(image), type(mask), type(image)].New()

    mf.SetInput1(image)
    mf.SetInput2(mask)
    mf.Update()
    return mf.GetOutput()



def get_sub_seg(n):

    path = n.replace("/T1/", "/T1_seg/")
    print(path)
    mask = itk.imread(path)

    mask = itk.CastImageFilter[type(mask), itk.Image[itk.SS, 3]].New()(mask)

    print(type(mask))
    
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
