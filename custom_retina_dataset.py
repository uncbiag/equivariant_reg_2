import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import tqdm

from icon_registration import config
def get_dataset_retina(
    extra_deformation=False,
    downsample_factor=4,
    blur_sigma=None,
    warps_per_pair=20,
    fixed_vertical_offset=None,
    always_offset_bottom=False,
    include_boundary=False,
    scale=None,
    split="train"
):
    try:
        import elasticdeform
        import hub
    except:

        raise Exception(
            """the retina dataset requires the dependencies hub and elasticdeform.
            Try pip install hub elasticdeform"""
        )

    ds_name = f"dataset_cache/retina___{extra_deformation}{downsample_factor}{blur_sigma}{warps_per_pair}{fixed_vertical_offset}{include_boundary}{scale}{split}{always_offset_bottom}.trch"

    import os

    if os.path.exists(ds_name):
        augmented_ds1_tensor, augmented_ds2_tensor = torch.load(ds_name)
    else:

        res = []
        for batch in hub.load("hub://activeloop/drive-train").pytorch(
            num_workers=0, batch_size=4, shuffle=False
        ):
            if include_boundary:
                res.append(batch["manual_masks/mask"] ^ batch["masks/mask"])
            else:
                res.append(batch["manual_masks/mask"])


        res = torch.cat(res)
        ds_tensor = res[:, None, :, :, 0] * -1.0 + (not include_boundary)
        if split == "test":
            ds_tensor = ds_tensor[:3]
        else:
            ds_tensor = ds_tensor[3:]

        ds_tensor = torch.cat([ds_tensor, torch.flip(ds_tensor, [2])], dim=0)

        if fixed_vertical_offset is not None:
            if always_offset_bottom:
                ds2_tensor = torch.cat(
                    [ds_tensor, torch.zeros(ds_tensor.shape[0], 1, fixed_vertical_offset, 565)], axis=2
                )
            else:
                ds2_tensor = torch.cat(
                    [torch.zeros(ds_tensor.shape[0], 1, fixed_vertical_offset, 565), ds_tensor], axis=2
                )
                
            ds1_tensor = torch.cat(
                [ds_tensor, torch.zeros(ds_tensor.shape[0], 1, fixed_vertical_offset, 565)], axis=2
            )
        else:
            ds2_tensor = ds_tensor
            ds1_tensor = ds_tensor

        warped_tensors = []
        print("warping images to generate dataset")
        for _ in tqdm.tqdm(range(warps_per_pair)):
            ds_2_list = []
            for el in ds2_tensor:
                case = el[0]
                # TODO implement random warping on gpu
                case_warped = np.array(case)
                if extra_deformation:
                    case_warped = elasticdeform.deform_random_grid(
                        case_warped, sigma=60, points=3
                    )
                case_warped = elasticdeform.deform_random_grid(
                    case_warped, sigma=25, points=3
                )

                case_warped = elasticdeform.deform_random_grid(
                    case_warped, sigma=12, points=6, zoom=scale
                )
                ds_2_list.append(torch.tensor(case_warped)[None, None, :, :])
                ds_2_tensor = torch.cat(ds_2_list)
            warped_tensors.append(ds_2_tensor)

        augmented_ds2_tensor = torch.cat(warped_tensors)
        augmented_ds1_tensor = torch.cat([ds1_tensor for _ in range(warps_per_pair)])

        torch.save((augmented_ds1_tensor, augmented_ds2_tensor), ds_name)

    batch_size = 10
    import torchvision.transforms.functional as Fv

    if blur_sigma is None:
        ds1 = torch.utils.data.TensorDataset(
            F.avg_pool2d(augmented_ds1_tensor, downsample_factor)
        )
    else:
        ds1 = torch.utils.data.TensorDataset(
            Fv.gaussian_blur(
                F.avg_pool2d(augmented_ds1_tensor, downsample_factor),
                4 * blur_sigma + 1,
                blur_sigma,
            )
        )
    d1 = torch.utils.data.DataLoader(
        ds1,
        batch_size=batch_size,
        shuffle=False,
    )
    if blur_sigma is None:
        ds2 = torch.utils.data.TensorDataset(
            F.avg_pool2d(augmented_ds2_tensor, downsample_factor)
        )
    else:
        ds2 = torch.utils.data.TensorDataset(
            Fv.gaussian_blur(
                F.avg_pool2d(augmented_ds2_tensor, downsample_factor),
                4 * blur_sigma + 1,
                blur_sigma,
            )
        )

    d2 = torch.utils.data.DataLoader(
        ds2,
        batch_size=batch_size,
        shuffle=False,
    )

    return d1, d2
