import itk

import torch

out_list = []
import os
import footsteps
import tqdm
import numpy as np

footsteps.initialize()

for i in tqdm.tqdm(range(800)):
    path = f"/playpen-raid1/tgreer/equivariant_reg_2/results/ixi_preprocess-5/train/cropped_imgs/Case_{i:05}_0000.nii.gz"
    mask = f"/playpen-raid1/tgreer/equivariant_reg_2/results/ixi_preprocess-5/train/cropped_masks/Case_{i:05}_0000.nii.gz"
    if os.path.exists(path):
        img = itk.imread(path)
        mask = itk.imread(mask)
        out_list.append(
                torch.tensor(np.array(img)).float()[None, None] *
                torch.tensor(np.array(mask)).float()[None, None] )

torch.save(out_list, footsteps.output_dir + "train_imgs_tensor.trch")
