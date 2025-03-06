import SimpleITK
import numpy as np
import tqdm

root_path = "/playpen-raid1/Data/Totalsegmentator_dataset/"

import os

ct_names = os.listdir(root_path)
print(ct_names[0])

dataset = []

#dataset_schema: 
#[
#    {
#        "image": "/playpen-raid1/Data/Totalsegmentator_dataset/s0001/ct.nii.gz",
#        "present_modalities":set(["brain", "rib"])
#        
#    }
#]


for ct_name in tqdm.tqdm(ct_names):
    if  "meta" in ct_name:
        continue
    image_path = root_path + ct_name + "/ct.nii.gz"
    image = SimpleITK.ReadImage(image_path) 

    segmentations = os.listdir(root_path + ct_name + "/segmentations")
    valid_segmentations = set()
    for seg in segmentations:
        seg_a = SimpleITK.ReadImage(root_path + ct_name + "/segmentations/" + seg)
        seg_a = SimpleITK.GetArrayFromImage(seg_a)
        if np.max(seg_a):
            valid_segmentations.add(seg)
    dataset.append({
        "image": image_path,
        "present_modalities": valid_segmentations
})
import pickle
import footsteps
print(dataset)
pickle.dump(dataset, open(footsteps.output_dir + "dataset.pickle", "wb"))




