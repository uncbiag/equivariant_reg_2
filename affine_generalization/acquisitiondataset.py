import torch
import re as regex
import collections
import tqdm
import random
import glob
import footsteps

class Dataset:
    def __init__(self, input_shape, name: str, image_glob: str, cache_filename = None, maximum_images=None, match_regex=None, acquisition_match_regex=None, is_mask_regex=None):
        print(name)
        self.name = name
        self.image_glob = image_glob
        self.input_shape = input_shape
        self.match_regex = match_regex
        self.acquisition_match_regex = acquisition_match_regex
        self.is_mask_regex = is_mask_regex



        if not cache_filename:

            preliminary_acquisition_keys = set()
            #Get a list of aquisitions. Trim it if necessary
            paths = self.get_image_paths()
            patients = set()
            for key in paths:
                    if self.match_regex:
                        patient = regex.search(match_regex, key).group(1)
                    else:
                        patient = "singleton"
                    patients.add(patient)


            breaker = False

            for patient in patients:
                for key in paths:
                    if not(match_regex) or regex.search(match_regex, key).group(1) == patient:

                        if self.acquisition_match_regex:
                            acquisition = regex.search(acquisition_match_regex, key).group(1)
                        else:
                            acquisition = key
                        preliminary_acquisition_keys.add(acquisition)
                        if len(preliminary_acquisition_keys) >= maximum_images :
                            breaker = True
                if breaker:
                    break


            preliminary_acquisition_keys = list(preliminary_acquisition_keys)

            self.store = {}
            for acq_key in tqdm.tqdm(preliminary_acquisition_keys):

                acq_paths = [p for p in paths if regex.search(acquisition_match_regex, p).group(1) == acq_key]
                self.preprocess_itk_image(acq_paths)
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
            #paths = self.get_image_paths()
            self.store = loaded_cache["store"]
            #assert(paths[0] in self.store) # sanity check
        self.keys = list(self.store.keys())

        self.patient_lookup = collections.defaultdict(lambda: [])
        self.patient_keys = {}

        for key in self.store.keys():
            if self.match_regex:
                patient_key = regex.search(match_regex, key).group(1)
            else:
                patient_key = "singleton"
            self.patient_keys[key] = patient_key
            self.patient_lookup[patient_key].append(key)

        self.acquisition_lookup = collections.defaultdict(lambda: [])
        self.acquisition_keys = {}

        for key in self.store.keys():
            if self.acquisition_match_regex:
                acquisition = regex.search(acquisition_match_regex, key).group(1)
            else:
                acquisition = key
            self.acquisition_keys[key] = acquisition
            self.acquisition_lookup[acquisition].append(key)

    
    def __repr__(self):
        # Get all attributes of the instance
        attrs = self.__dict__
        
        # Format each attribute as key=value, excluding methods and private attributes
        attr_strs = [f"{key}={repr(value)}" for key, value in attrs.items() 
                     if not key.startswith('_') and not callable(value) and not key == "store"]
        
        # Create the representation string
        class_name = self.__class__.__name__
        return f"{class_name}(\n" + ",\n".join(attr_strs) + "\n)"
                

    def get_image_paths(self) -> [str]:
        return list(glob.glob(self.image_glob))
    
    def read_image(self, path:str):
        import SimpleITK
        image = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(path))
        image = torch.tensor(image)
        return image


    def preprocess_itk_image(self, paths:[str]):
        """
        Load an image, crop away any black bars, and then resize to the target resolution.
        """
        raw_images = []
        for path in paths:

            image = self.read_image(path) # factored into a function for DICOMDataset to override
            original_type = image.dtype
            image = image[None, None]
            raw_images.append(image.float())

        catted_images = torch.cat(raw_images)
        catted_images = torch_crop_foreground(catted_images, additional_crop_pixels=4)

        
        for i, path in enumerate(paths):

            image = catted_images[i:i + 1]
            if not(self.is_mask_regex) or not(regex.match(self.is_mask_regex, path)):
                # Not a mask 
                image = image.float()
                image = torch.nn.functional.interpolate(image, self.input_shape[2:], mode="trilinear")
                im_min, im_max = (
                        torch.quantile(image.view(-1), 0.01),
                        torch.quantile(image.view(-1), 0.99)
                        )
                image = torch.clip(image, im_min, im_max)
                image = image - im_min
                image = image / (im_max - im_min)
                image = image * 255.
                self.store[path] = image.to(torch.uint8)
            else:
                image = image.uint16()
                image = torch.nn.functional.interpolate(image, self.input_shape[2:], mode="nearest")
                self.store[path] = image





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
        unprepped_image = unprepped_image.float()
        unprepped_image = unprepped_image - torch.min(unprepped_image)
        unprepped_image = unprepped_image.float() / torch.max(unprepped_image)
        return unprepped_image # not really unprepped anymore

    def get_keys_same_patient(self) -> tuple[str, str]:
        """
        get a pair of images from the dataset from the same patient
        """
        
        image_key_1 = random.choice(self.keys)
        image_key_2 = image_key_1
        count = 0
        while image_key_2 == image_key_1:
            image_key_2 = random.choice(self.patient_lookup[self.patient_keys[image_key_1]])
            count += 1
            if count > 8:
                return self.get_key_patient()
        return (image_key_1, image_key_2)
    def get_pair(self):
        pair = self.get_keys_same_patient()
        return self.get_image(pair[0]), self.get_image(pair[1])






class DICOMDataset(Dataset):
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


