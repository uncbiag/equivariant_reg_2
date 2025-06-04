import itk

import matplotlib.pyplot as plt
import numpy as np
import icon_registration.itk_wrapper

moving = itk.imread("register_py/bonefrags/bone_4.nrrd")
fixed = itk.imread("register_py/bonefrags/bone_0.nrrd")

phi_AB = itk.transformread("trans.hdf5")[0]

voxels = 160

def decompose_icon_itk_transform(phi_AB:itk.CompositeTransform):

    original_displacement_transform = phi_AB.GetNthTransform(1)
    original_displacement_transform = itk.DisplacementFieldTransform[itk.D, 3].cast(original_displacement_transform)
    displacement_image = original_displacement_transform.GetDisplacementField()
    original_displacement_array = itk.GetArrayFromImage(displacement_image)

    coordinates = np.mgrid[0:voxels, 0:voxels, 0:voxels]
    coordinates = coordinates.transpose((3, 2, 1, 0))
    coordinates = np.concatenate([coordinates, np.ones((voxels, voxels, voxels, 1))], axis=-1)

    x = coordinates.reshape(-1, 4)
    y = original_displacement_array.reshape(-1, 3)

    best_affine_fit = np.linalg.inv(x.T @ x) @ (x.T @ y)

    error =  (y - x @ best_affine_fit)

    Offset = best_affine_fit[3]

    Matrix = best_affine_fit[:3].transpose() + np.eye(3)


    error = error @ np.linalg.inv(Matrix.transpose())
    error = error.reshape(voxels, voxels, voxels, 3)
    residual_displacement_transform = itk.DisplacementFieldTransform[(itk.D, 3)].New()
    itk_disp_field = itk.image_from_array(error, is_vector=True)
    residual_displacement_transform.SetDisplacementField(itk_disp_field)

    transformType = itk.CenteredAffineTransform[itk.D, 3]
    affine_component_of_network_transform = transformType.New()
    affine_component_of_network_transform.SetOffset(Offset)
    affine_component_of_network_transform.SetCenter((0, 0, 0))
    affine_component_of_network_transform.SetMatrix(itk.matrix_from_array(Matrix))

    affine_decomposed_transform = itk.CompositeTransform[itk.D, 3].New()

    affine_decomposed_transform.PrependTransform(phi_AB.GetNthTransform(2))
    affine_decomposed_transform.PrependTransform(residual_displacement_transform)
    affine_decomposed_transform.PrependTransform(affine_component_of_network_transform)
    affine_decomposed_transform.PrependTransform(phi_AB.GetNthTransform(0))
    return affine_decomposed_transform

def extract_icon_itk_transform(phi_AB):
    phi_AB = decompose_icon_itk_transform(phi_AB)

    affine_Transform = itk.CompositeTransform[itk.D, 3].New()

    affine_Transform.PrependTransform(phi_AB.GetNthTransform(3))
    affine_Transform.PrependTransform(phi_AB.GetNthTransform(1))
    affine_Transform.PrependTransform(phi_AB.GetNthTransform(0))

    return affine_Transform



from register import torch, quantile
def preprocess(image):



    image = itk.CastImageFilter[type(image), itk.Image[itk.F, 3]].New()(image)
    min_ = quantile(torch.tensor(np.array(image)), .01).item()
    max_ = quantile(torch.tensor(np.array(image)), .99).item()
    image = itk.clamp_image_filter(image, Bounds=(min_, max_))

    image = itk.shift_scale_image_filter(image, shift=-min_, scale = 1/(max_-min_)) 

    return image

if __name__ == "__main__":
    phi_AB_decomposed = decompose_icon_itk_transform(phi_AB)
    itk.transformwrite([phi_AB_decomposed], "decomposed.hdf5")


    moving = itk.CastImageFilter[type(moving), itk.Image[itk.F, 3]].New()(moving)
    interpolator = itk.LinearInterpolateImageFunction.New(moving)
    warped_moving_image = itk.resample_image_filter(
            moving,
            transform=phi_AB,
            interpolator=interpolator,
            use_reference_image=True,
            reference_image=fixed
            )
    interpolator_d = itk.LinearInterpolateImageFunction.New(moving)
    warped_moving_image_d = itk.resample_image_filter(
            moving,
            transform=phi_AB_decomposed,
            interpolator=interpolator_d,
            use_reference_image=True,
            reference_image=fixed
            )

    plt.clf()

    plt.ion()
    print(fixed.shape)

    slice_ = 80

    for _ in range(0):
        plt.imshow(np.minimum(10000, -np.array(preprocess(warped_moving_image_d)[:, slice_])), cmap="Grays")
        plt.show()
        plt.pause(.9)

        plt.imshow(np.minimum(10000, -np.array(preprocess(warped_moving_image)[:, slice_])), cmap="Grays")
        plt.show()
        plt.pause(.9)

    plt.imshow(np.minimum(10000, -np.array(itk.checker_board_image_filter(preprocess(warped_moving_image_d), preprocess(fixed))[:, slice_])), cmap="Grays")
    #plt.imshow(np.minimum(100, -np.original_displacement_array(preprocess(warped_moving_image)[:, 50])), cmap="Grays")
    #plt.imshow(np.minimum(100, -np.original_displacement_array(preprocess(fixed)[:, 50])), cmap="Grays")
    plt.show()
    plt.pause(.4)
    #plt.show()

