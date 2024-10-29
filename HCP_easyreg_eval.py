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





def get_unstripped_brain_image_path(n):
    return f"/playpen-raid2/Data/HCP/HCP_1200/{n}/T1w/T1w_acpc_dc_restore.nii.gz"



dices = []
flips = []
ICON_errors=[]

import glob

from HCP_segs import atlas_registered, get_brain_image, get_sub_seg_path

random.seed(1)
for _ in range(100):
    n_A, n_B = (random.choice(atlas_registered) for _ in range(2))
    image_A, image_B = ((get_unstripped_brain_image_path(n)) for n in (n_A, n_B))


    segmentation_A, segmentation_B = ((get_sub_seg_path(n)) for n in (n_A, n_B))

    image_A = itk.imread(image_A)
    segmentation_A = itk.imread(segmentation_A)

    args = lambda: None

    args.slide = True
    args.transpose = False

    if args.slide:
        image_A_buffer = itk.PyBuffer[type(image_A)].GetArrayViewFromImage(image_A)
        image_A_buffer[:] = np.roll(image_A_buffer,45)

        segmentation_A_buffer = itk.PyBuffer[type(segmentation_A)].GetArrayViewFromImage(segmentation_A)
        segmentation_A_buffer[:] = np.roll(segmentation_A_buffer, 45)

    if args.transpose:
        image_A_buffer = itk.PyBuffer[type(image_A)].GetArrayViewFromImage(image_A)
        image_A_buffer[:] = np.rot90(image_A_buffer, axes=(0, 1))

        segmentation_A_buffer = itk.PyBuffer[type(segmentation_A)].GetArrayViewFromImage(segmentation_A)
        segmentation_A_buffer[:] = np.rot90(segmentation_A_buffer, axes=(0, 1))
    data_path = footsteps.output_dir   + str(_) + "/"

    os.makedirs(data_path)

    itk.imwrite(image_A, f"{data_path}{n_A}.nii.gz")
    image_A = f"{data_path}{n_A}.nii.gz"

    itk.imwrite(segmentation_A, f"{data_path}{n_A}_seg.nii.gz")
    segmentation_A = f"{data_path}{n_A}_seg.nii.gz"


    import sys

    import subprocess
    register_command = f"mri_easyreg --ref {image_B} --flo {image_A} --ref_reg {data_path}{n_B}_B_warped.nii.gz --flo_reg {data_path}{n_A}_A_warped.nii.gz --fwd_field {data_path}fwd.nii.gz --ref_seg {data_path}{n_B}_B_seg.nii.gz --flo_seg {data_path}{n_A}_A_seg.nii.gz --threads 12"

    print(register_command)

    subprocess.run(register_command, shell=True)

    warp_command = f"mri_easywarp --i {segmentation_A} --o {data_path}warped_segmentation_A.nii.gz --field {data_path}fwd.nii.gz --threads 12 --nearest"

    print(warp_command)

    subprocess.run(warp_command, shell=True)

    segmentation_B = itk.imread(segmentation_B)

    warped_segmentation_A = itk.imread(f"{data_path}warped_segmentation_A.nii.gz")

    mean_dice = utils.itk_mean_dice(segmentation_B, warped_segmentation_A)

    utils.log(_)
    utils.log(n_A, n_B)
    utils.log(mean_dice)

    dices.append(mean_dice)

utils.log("Mean DICE")
utils.log(np.mean(dices))
