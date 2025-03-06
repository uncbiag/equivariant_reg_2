import torch
import re as regex
import collections
import tqdm
import random
import glob
import footsteps

class Dataset:
    def __init__(self, input_shape, name: str, image_glob: str, cache_filename = None, maximum_images=None):
        print(name)
        self.name = name
        self.image_glob = image_glob
        self.input_shape = input_shape

        if not cache_filename:
            self.store = {}
            paths = self.get_image_paths()
            if maximum_images:
                paths = paths[:maximum_images]
            for path in tqdm.tqdm(paths):
                self.store[path] = self.preprocess_itk_image(path)
            torch.save(
                    {
                        "name": self.name,
                        "image_glob": self.image_glob,
                        "maximum_images": maximum_images,
                        "store": self.store
                    },
                    footsteps.output_dir + self.name + "_cached_dataset.trch")
        else:
            loaded_cache = torch.load(cache_filename + "/" + self.name + "_cached_dataset.trch")
            assert(self.name == loaded_cache["name"])
            assert(maximum_images == loaded_cache["maximum_images"])
            assert(self.image_glob == loaded_cache["image_glob"])
            paths = self.get_image_paths()
            self.store = loaded_cache["store"]
            assert(paths[0] in self.store) # sanity check
        self.keys = list(self.store.keys())
                

    def get_image_paths(self) -> [str]:
        return list(glob.glob(self.image_glob))
    
    def read_image(self, path:str):
        import SimpleITK
        image = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(path))
        image = torch.tensor(image)
        return image


    def preprocess_itk_image(self, path:str):
        """
        Load an image, crop away any black bars, and then resize to the target resolution.
        """
        image = self.read_image(path) # factored into a function for DICOMDataset to override
        original_type = image.dtype
        image = image[None, None]
        image = torch_crop_foreground(image, additional_crop_pixels=4)
        image = image.float()
        image = torch.nn.functional.interpolate(image, self.input_shape[2:], mode="trilinear")
        im_min, im_max = (
                torch.quantile(image.view(-1), 0.01),
                torch.quantile(image.view(-1), 0.99)
                )
        image = torch.clip(image, im_min, im_max)
        return image.type(original_type)



    def get_image(self, key: str) -> torch.Tensor:
        """
        returns an image from the dataset. 
        the image is stored compressed and potentially as an unnormalized int,
        and this method converts it to [0, 1] normalized cuda float.
        The images should be resized to the standard size before putting them
        into the store, and have any black bars trimmed off.
        the images are stored in [B, C, H, W, D] with the first 2 channels length 1
        The images should have their 99th percentile clippe
        d before putting them into the store.
        """
        unprepped_image = self.store[key]
        unprepped_image = unprepped_image.cuda().float()
        unprepped_image = unprepped_image - torch.min(unprepped_image)
        unprepped_image = unprepped_image.float() / torch.max(unprepped_image)
        return unprepped_image # not really unprepped anymore

    def get_key_pair(self) -> tuple[str, str]:
        """
        get a pair of images from the dataset. 
        This is the one that should be overridden to differentiate between
        paired and unpaired datasets.
        """
        return (random.choice(self.keys), random.choice(self.keys))
    def get_pair(self):
        pair = self.get_key_pair()
        return self.get_image(pair[0]), self.get_image(pair[1])


class PairedDataset(Dataset):
    def __init__(self, input_shape, name: str, image_glob: str, cache_filename = None, maximum_images=None, match_regex=None):
        super().__init__(input_shape, name, image_glob, cache_filename=cache_filename, maximum_images=maximum_images)
        if match_regex == None:
            raise NotImplementedError()

        self.pair_lookup = collections.defaultdict(lambda: [])
        self.pair_keys = {}

        for key in self.store.keys():
            pair_key = regex.search(match_regex, key).group(1)
            self.pair_keys[key] = pair_key
            self.pair_lookup[pair_key].append(key)
        for pair_key in self.pair_lookup.keys():
            assert len(self.pair_lookup[pair_key]) != 1 , f"{self.pair_lookup[pair_key]}"



    def get_key_pair(self):
        image_key_1 = random.choice(self.keys)
        image_key_2 = image_key_1
        while image_key_2 == image_key_1:
            image_key_2 = random.choice(self.pair_lookup[self.pair_keys[image_key_1]])
        return (image_key_1, image_key_2)

class PairedDICOMDataset(PairedDataset):
    def read_image(self, path:str):
      """
      Reads a DICOM series from a directory path and returns it as a tensor.
      
      Args:
         path (str): Directory containing DICOM files
            e.g., "files/image342/"
      
      Returns:
         torch.Tensor: 3D tensor containing the DICOM volume
      """
      import SimpleITK as sitk
      import os
      print(os.listdir(path))
      # Get all DICOM files in the directory
      dicom_files = [os.path.join(path, f) for f in os.listdir(path) 
                 if os.path.isfile(os.path.join(path, f)) and f.endswith('.dicom')]
      
      # Sort the files to ensure correct slice order
      dicom_files.sort()
      
      if not dicom_files:
         raise ValueError(f"No DICOM files found in directory: {path}")
      
      # Read the DICOM series as a 3D image
      reader = sitk.ImageSeriesReader()
      reader.SetFileNames(dicom_files)
      image = reader.Execute()
      
      # Convert to tensor
      image_array = sitk.GetArrayFromImage(image)
      image_tensor = torch.tensor(image_array)
      
      return image_tensor




def torch_crop_foreground(tensor: torch.Tensor, additional_crop_pixels: int = 0) -> torch.Tensor:
    """
    Crops a PyTorch tensor to its foreground by removing uniform boundary regions.
    
    This function finds the first non-uniform slice from each direction and crops the tensor
    accordingly. It works with both 2D and 3D tensors.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (B, C, H, W) for 2D or (B, C, D, H, W) for 3D
        additional_crop_pixels (int, optional): Additional pixels to crop from each boundary. 
            Defaults to 0.
            
    Returns:
        torch.Tensor: Cropped tensor containing only the foreground region
        
    Raises:
        ValueError: If input tensor is not 2D or 3D
        
    Example:
        >>> x = torch.zeros((100, 100))
        >>> x[25:75, 25:75] = 1
        >>> cropped = torch_crop_foreground(x[None, None])
        >>> print(cropped.shape)
        torch.Size([1, 1, 50, 50])
    """
    if not (2 <= tensor.dim() - 2 <= 3):
        raise ValueError("Input tensor must be 2D or 3D")
        
    def first_nonequal(fn):
        i = 0
        while True:
            slice_tensor = fn(i)
            if not torch.all(slice_tensor == slice_tensor.flatten()[0]):
                return i + additional_crop_pixels
            i += 1
            
    if tensor.dim() - 2 == 2:
        # Find boundaries for 2D tensor
        upper_1 = first_nonequal(lambda i: tensor[:, :, :, tensor.shape[3] - 1 - i])
        upper_2 = first_nonequal(lambda i: tensor[:, :, tensor.shape[2] - 1 - i])
        
        lower_1 = first_nonequal(lambda i: tensor[:, :, :, i])
        lower_2 = first_nonequal(lambda i: tensor[:, :, i])
        
        # Crop the tensor
        return tensor[:, :, lower_2:tensor.shape[2] - upper_2, 
                     lower_1:tensor.shape[3] - upper_1]
    
    else:  # 3D case
        # Find boundaries for 3D tensor
        upper_1 = first_nonequal(lambda i: tensor[:, :, :, :, tensor.shape[4] - 1 - i])
        upper_2 = first_nonequal(lambda i: tensor[:, :, :, tensor.shape[3] - 1 - i])
        upper_3 = first_nonequal(lambda i: tensor[:, :, tensor.shape[2] - 1 - i])
        
        lower_1 = first_nonequal(lambda i: tensor[:, :, :, :, i])
        lower_2 = first_nonequal(lambda i: tensor[:, :, :, i])
        lower_3 = first_nonequal(lambda i: tensor[:, :, i])
        
        # Crop the tensor
        return tensor[:, :, lower_3:tensor.shape[2] - upper_3,
                     lower_2:tensor.shape[3] - upper_2,
                     lower_1:tensor.shape[4] - upper_1]


