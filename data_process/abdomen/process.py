import itk
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
import footsteps
import tqdm

def resample(img, crop=False):
    new_spacing = np.array(img.GetSpacing()) * np.array(img.GetLargestPossibleRegion().GetSize()) / np.array(target_size)
    
    size = itk.Size[3]()
    for i in range(3):
        size.SetElement(i, target_size[i])

    if crop:
        new_spacing[-1] = new_spacing[-2]
    resampler = itk.ResampleImageFilter.New(Input=img)
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
            img = itk.imread(glob.glob(f"/playpen-raid2/Data/AbdomenCT-1K/AbdomenCT-1K-ImagePart*/Case_00{i:03}_0000.nii.gz")[0])
            mask = itk.imread(f"/playpen-raid2/Data/AbdomenCT-1K/Mask/Case_00{i:03}.nii.gz")
            
            out, outmask = [resample(a) for a in [img, mask]]

            itk.imwrite(out, f"{path}stretched_imgs/Case_{i:05}_0000.nii.gz") 
            itk.imwrite(outmask, f"{path}stretched_masks/Case_{i:05}_0000.nii.gz") 
            
            #plt.imshow(out[:, 90])
            #plt.show()
            #plt.imshow(outmask[:, 90])
            
            #plt.imshow(np.max(out, axis=1))
            #plt.show()
            #plt.imshow(np.max(outmask, axis=1))
            #plt.show()
            
            out, outmask = [resample(a, crop=True) for a in [img, mask]]
            itk.imwrite(out, f"{path}cropped_imgs/Case_{i:05}_0000.nii.gz") 
            itk.imwrite(outmask, f"{path}cropped_masks/Case_{i:05}_0000.nii.gz") 
            
            #plt.imshow(out[:, 90])
            #plt.show()
            #plt.imshow(outmask[:, 90])
            
            #plt.imshow(np.max(out, axis=1))
            #plt.show()
            #plt.imshow(np.max(outmask, axis=1))
            #plt.show()
        except :
            pass

import footsteps

process(range(1, 800), footsteps.output_dir + "train/")
process(range(800, 900), footsteps.output_dir + "val/")
process(range(900, 1000), footsteps.output_dir + "test/")
