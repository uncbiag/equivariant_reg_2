import itk
import numpy as np


def itk_crop_foreground(image, additional_crop_pixels=0):

    def first_nonequal(fn):
        i = 0
        while True:
            slice = fn(i)
            if not np.all(slice == slice.flat[0]):
                return i + additional_crop_pixels
            i += 1

    arr = itk.GetArrayFromImage(image)

    
    if len(arr.shape) == 2:
        
        upper_1 = first_nonequal(lambda i: arr[:, arr.shape[1] - 1 - i])
        upper_2 = first_nonequal(lambda i: arr[arr.shape[0] - 1 - i])
    
        upper = (upper_1, upper_2)
    
        lower_1 = first_nonequal(lambda i: arr[:, i])
        lower_2 = first_nonequal(lambda i: arr[i])
    
        lower = (lower_1, lower_2)
    if len(arr.shape) == 3:
        upper_1 = first_nonequal(lambda i: arr[:, :, arr.shape[2] - 1 - i])
        upper_2 = first_nonequal(lambda i: arr[:, arr.shape[1] - 1 - i])
        upper_3 = first_nonequal(lambda i: arr[arr.shape[0] - 1 - i])
    
        upper = (upper_1, upper_2, upper_3)
        
        lower_1 = first_nonequal(lambda i: arr[:, :, i])
        lower_2 = first_nonequal(lambda i: arr[:, i])
        lower_3 = first_nonequal(lambda i: arr[i])
    
        lower = (lower_1, lower_2, lower_3)

    print(lower, upper)

        
    
    crop_filter = itk.CropImageFilter[type(image), type(image)].New()
    crop_filter.SetUpperBoundaryCropSize(upper)
    crop_filter.SetLowerBoundaryCropSize(lower)
    crop_filter.SetInput(image)
    crop_filter.Update()

    output = crop_filter.GetOutput()

    itk.imwrite(output, "/playpen/tgreer/tmp_img.nrrd")
    output = itk.imread("/playpen/tgreer/tmp_img.nrrd")

    return output

