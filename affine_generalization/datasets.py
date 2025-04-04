import dataset
import torch
import footsteps
import matplotlib.pyplot as plt

input_shape = (1, 1, 160, 160, 160)


cache_filename = "results/8bit_dataset/"
#cache_filename=None
maximum_images=1000

#cache_filename = "results/dp_r"
#maximum_images = 4

datasets_ = []
datasets_.append(dataset.Dataset(input_shape, "abdomen8k", "/data/hastings/Abdomen8k/AbdomenAtlas/*/ct.nii.gz", maximum_images=maximum_images, cache_filename="results/-4/"))
datasets_.append(dataset.Dataset(input_shape, "TotalSegmentatorMRI", "/playpen-raid1/soumitri/data/TotalSegMRI/*/*/mri.nii.gz", maximum_images=maximum_images, cache_filename="results/-4/"))

datasets_.append(dataset.PairedDataset(input_shape, "bratsreg", "/playpen-raid2/Data/BraTS-Reg/BraTSReg_Training_Data_v3/*/*.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename, match_regex=r"v3/(BraTSReg_[0-9]+)/"))
datasets_.append(dataset.PairedDataset(input_shape, "pancreas", "/playpen-raid1/tgreer/pancreatic_cancer_registration/data/*/Processed/*/original_image.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename, match_regex=r"data/([0-9]+)/Processed/"))
datasets_.append(dataset.PairedDataset(input_shape, "dirlab_clamped", "/playpen-raid2/Data/Lung_Registration_clamp_normal_transposed/*/*_img.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename, match_regex=r'transposed/([a-zA-Z0-9]+)/'))
datasets_.append(dataset.PairedDataset(input_shape, "dirlab", "/playpen-raid2/Data/Lung_Registration_transposed/*/*_img.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename, match_regex=r'transposed/([a-zA-Z0-9]+)/'))

datasets_.append(dataset.Dataset(input_shape, "abdomen1k", "/playpen-raid2/Data/AbdomenCT-1K/AbdomenCT-1K-ImagePart*/Case_*", maximum_images=maximum_images, cache_filename=cache_filename))
datasets_.append(dataset.Dataset(input_shape, "HCP_t1_stripped", "/playpen-raid2/Data/HCP/HCP_1200/*/T1w/T1w_acpc_dc_restore_brain.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename))
datasets_.append(dataset.Dataset(input_shape, "HCP_t1", "/playpen-raid2/Data/HCP/HCP_1200/*/T1w/T1w_acpc_dc_restore.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename))
datasets_.append(dataset.Dataset(input_shape, "HCP_t2_stripped", "/playpen-raid2/Data/HCP/HCP_1200/*/T1w/T2w_acpc_dc_restore_brain.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename))
datasets_.append(dataset.Dataset(input_shape, "HCP_t2", "/playpen-raid2/Data/HCP/HCP_1200/*/T1w/T2w_acpc_dc_restore.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename))
datasets_.append(dataset.Dataset(input_shape, "OAI", "/playpen-raid/zhenlinx/Data/OAI_segmentation/Nifti_rescaled_LEFT/*_image.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename))
datasets_.append(dataset.Dataset(input_shape, "fmost", "/playpen-raid2/Data/fMost/subject/*_red_mm_RSA.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename))
###datasets_.append(dataset.Dataset(input_shape, "heart", "/playpen-raid/bdemir/uniICON/ACDC/database/training/*/patient003_frame15.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename))
###datasets_.append(dataset.Dataset(input_shape, "ixi", "/playpen-raid2/Data/IXI/IXI_data/Train/*", maximum_images=maximum_images, cache_filename=cache_filename))
datasets_.append(dataset.Dataset(input_shape, "oasis", "/playpen-raid2/Data/oasis/OASIS_OAS1_*_MR1/orig.nii.gz", maximum_images=maximum_images, cache_filename=cache_filename))
datasets_.append(dataset.Dataset(input_shape, "lumir", "/playpen-raid1/Data/LUMIR/imagesTr/*", maximum_images=maximum_images, cache_filename=cache_filename))


datasets_.append(dataset.PairedDICOMDataset(input_shape, "DukeLivers", "/playpen-raid1/Data/DukeLivers/Segmentation/Segmentation/*/*/images/", maximum_images=maximum_images, match_regex=r"Segmentation/([0-9]+)/", cache_filename=cache_filename))
datasets_.append(dataset.Dataset(input_shape, "translucence", "/playpen-raid1/tgreer/mouse_brain_translucence/data/auto_files_resampled/*", cache_filename=cache_filename, maximum_images=maximum_images))
maximum_images=20000
cache_filename="results/CPTAC-12/"
datasets_.append(dataset.PairedDataset(input_shape, "autoPET", "/playpen1/tgreer/PET/FDG-PET-CT-Lesions/*/*/[PC][TE][Tr]*.nii.gz", match_regex=r"(/PETCT_[0-9a-z]+/)", maximum_images=maximum_images, cache_filename=cache_filename))
datasets_.append(dataset.PairedDICOMDataset(input_shape, "CPTAC-UCEC", "/playpen-raid1/Data/TCIA_CPTAC-UCEC/manifest-1712342731330/CPTAC-UCEC/*/*/*/", match_regex=r"/(C3[NL]-[0-9]*)/", maximum_images=maximum_images, cache_filename=cache_filename) )
maximum_images=1000

cache_filename="results/mrct_custom-8"
datasets_.append(dataset.PairedDICOMDataset(input_shape, "TCIA-hastings", "/playpen-raid1/Data/TCIA_Hastings_custom_mrct/manifest-1743108366953/*/*/*/*/", match_regex=r"66953/[a-zA-Z\-]*/([A-Z0-9\-]+)/", maximum_images=maximum_images, cache_filename=cache_filename))



if __name__ == "__main__":
    for d in datasets_:
        for i in range(1):
            pair = d.get_pair()
            plt.imshow(torch.cat(pair, dim=2)[0, 0, :, :, 50].cpu())
            footsteps.plot(d.name)
            plt.imshow(torch.max(torch.cat(pair, dim=2), dim=4).values[0, 0].cpu())
            footsteps.plot(d.name)
            plt.imshow(torch.cat(pair, dim=2)[0, 0, :, 50].cpu())
            footsteps.plot(d.name)
            plt.imshow(torch.cat(pair, dim=4)[0, 0, 50].cpu())
            footsteps.plot(d.name)
