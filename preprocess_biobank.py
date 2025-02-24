import footsteps
import torch
import footsteps
import torch
import itk
import tqdm
import numpy as np
import glob
import torch.nn.functional as F

input_shape=(1, 1, 64, 156, 224)

footsteps.initialize()

image_paths = glob.glob("/playpen-raid/bdemir/3dextension/regions_water/*.nii.gz") #

ds = []

def process(image):
    image = image[None, None] # add batch and channel dimensions

    #image = torch.nn.functional.avg_pool3d(image, 2)
    image = F.interpolate(image, input_shape[2:], mode="trilinear")
    print(image.shape)
    assert(image.shape == input_shape)

    return image

for name in tqdm.tqdm(list(iter(image_paths))[:]):

    image = torch.tensor(np.asarray(itk.imread(name)).astype(np.float32))

    ds.append(process(image))

torch.save(ds, f"{footsteps.output_dir}/training_data.trch")
