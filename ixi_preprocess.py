import itk
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import footsteps
import tqdm

def resample(img, crop=False, mask=False):
    new_spacing = np.array(img.GetSpacing()) * np.array(img.GetLargestPossibleRegion().GetSize()) / np.               array(target_size)

    size = itk.Size[3]()
    for i in range(3):
        size.SetElement(i, target_size[i])

    if crop:
        new_spacing[-1] = new_spacing[-2]

    resampler = itk.ResampleImageFilter.New(Input=img)

    if not mask:
        interpolator = itk.LinearInterpolateImageFunction.New(img)
        resampler.SetDefaultPixelValue(-1000)
    else:
        interpolator = itk.NearestNeighborInterpolateImageFunction.New(img)
    resampler.SetInterpolator(interpolator)
    resampler.SetSize(size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetTransform(itk.TranslationTransform[itk.D, 3].New())
    resampler.Update()

    return resampler.GetOutput()
    #target size 175 175 175
target_size = [175, 175, 175]

def process(range_, path):
    os.makedirs(path + "stretched_imgs/")
    os.makedirs(path + "stretched_masks/")
    os.makedirs(path + "cropped_imgs/")
    os.makedirs(path + "cropped_masks/")


    for i in tqdm.tqdm(range_):
        try:
            img = itk.imread(glob.glob(f"/playpen-raid1/tgreer/equivariant_reg_2/keymorph_comparison/keymorph/data/processed_IXI/T1/IXI{i:03}*.nii")[0])
            mask = itk.imread(glob.glob(f"/playpen-raid1/tgreer/equivariant_reg_2/keymorph_comparison/keymorph/data/processed_IXI/T1_mask/IXI{i:03}*")[0])

            print("success")
            out, outmask = resample(img, mask=False), resample(mask, mask=True)
            itk.imwrite(out, f"{path}stretched_imgs/Case_{i:05}_0000.nii.gz")
            itk.imwrite(outmask, f"{path}stretched_masks/Case_{i:05}_0000.nii.gz")

            #plt.imshow(out[:, 90])
            #plt.show()
            #plt.imshow(outmask[:, 90])

            #plt.imshow(np.max(out, axis=1))
            #plt.show()
            #plt.imshow(np.max(outmask, axis=1))
            #plt.show()

            out, outmask = resample(img, mask=False, crop=True), resample(mask, mask=True, crop=True)
            itk.imwrite(out, f"{path}cropped_imgs/Case_{i:05}_0000.nii.gz")
            itk.imwrite(outmask, f"{path}cropped_masks/Case_{i:05}_0000.nii.gz")

            #plt.imshow(out[:, 90])
            #plt.show()
            #plt.imshow(outmask[:, 90])

            #plt.imshow(np.max(out, axis=1))
            #plt.show()
            #plt.imshow(np.max(outmask, axis=1))
            #plt.show()
        except IndexError as e:
            print(e)

import footsteps

process(range(1, 428), footsteps.output_dir + "train/")
process(range(428, 1000), footsteps.output_dir + "test/")

