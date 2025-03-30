import dataset
import torch
import footsteps
import matplotlib.pyplot as plt

input_shape = (1, 1, 160, 160, 160)


cache_filename="results/mrct_custom-8"
maximum_images=1000


datasets_ = []
#datasets_.append(dataset.PairedDICOMDataset(input_shape, "CPTAC-UCEC", "/playpen-raid1/Data/TCIA_CPTAC-UCEC/manifest-1712342731330/CPTAC-UCEC/*/*/*/", match_regex=r"/(C3[NL]-[0-9]*)/", maximum_images=maximum_images, cache_filename=cache_filename) )
#datasets_.append(dataset.PairedDataset(input_shape, "autoPET", "/playpen1/tgreer/PET/FDG-PET-CT-Lesions/*/*/[PC][TE][Tr]*.nii.gz", match_regex=r"(/PETCT_[0-9a-z]+/)", maximum_images=maximum_images, cache_filename=cache_filename))

datasets_.append(dataset.PairedDICOMDataset(input_shape, "TCIA-hastings", "/playpen-raid1/Data/TCIA_Hastings_custom_mrct/manifest-1743108366953/*/*/*/*/", match_regex=r"66953/[a-zA-Z\-]*/([A-Z0-9\-]+)/", maximum_images=maximum_images, cache_filename=cache_filename))

print(datasets_[0].keys)


if __name__ == "__main__":
    for d in datasets_:
        for i in range(22):
            pair = d.get_pair()
            plt.imshow(torch.cat(pair, dim=2)[0, 0, :, :, 50].cpu())
            footsteps.plot("-----------------------" + d.name)
            plt.imshow(torch.max(torch.cat(pair, dim=2), dim=4).values[0, 0].cpu())
            footsteps.plot(d.name)
            plt.imshow(torch.cat(pair, dim=2)[0, 0, :, 50].cpu())
            footsteps.plot(d.name)
            plt.imshow(torch.cat(pair, dim=4)[0, 0, 50].cpu())
            footsteps.plot(d.name)
