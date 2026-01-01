import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import os
import sys
# Add the parent directory of folder1 to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Assuming 'image_tensor' is your tensor with shape [3, 256, 256]
def save_tensor_as_image(image_tensor, save_path="output.jpg"):
    # Undo normalization if it was applied
    unnormalize = transforms.Compose([
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # Undo normalization
    ])
    
    # Apply unnormalization
    image_tensor = unnormalize(image_tensor)

    # Convert tensor to a PIL Image
    to_pil = transforms.ToPILImage()
    image_pil = to_pil(image_tensor)

    # Save the image
    image_pil.save(save_path)
    print(f"Image saved at {save_path}")

# Example usage:
# Assuming 'image' is a tensor from your dataset
# image, _ = dataset[0]  # Get image tensor
# save_tensor_as_image(image, "output.jpg")

def RK4step(model, x, t, dt):
    k1 = model(x, t)
    k2 = model(x+dt/2*k1, t+dt/2)
    k3 = model(x+dt/2*k2, t+dt/2)
    k4 = model(x+dt*k3, t+dt)
    step = 1/6*(k1 + 2*k2 + 2*k3 + k4)
    return step

def RK4(model, x, dt, nt, traj=False):
    t = torch.zeros(x.shape[0]).to(x.device)
    if traj:
        X = [x]
    with torch.no_grad():
        for i in range(nt):
            step = RK4step(model, x, t, dt)
            x  = x + dt*step
            t = t + dt
            if traj:
                X.append(x)
            #print(t[0])
    if traj:
        return x,  torch.stack(X)
    else:
        return x

class AFHQDataset(Dataset):
    def __init__(self, root_dir, split="train", categories=["cat", "dog", "wild"]):
        """
        Args:
            root_dir (str): Root directory containing 'train' and 'val' folders.
            split (str): "train" or "val".
            categories (list): List of categories to include (e.g., ["cat", "dog"]).
        """
        assert split in ["train", "val"], "Split must be 'train' or 'val'"
        
        self.root_dir = root_dir
        self.split = split
        self.categories = categories

        # Define standard transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to 256x256
            # transforms.RandomHorizontalFlip(p=0.5),  # Data Augmentation
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
        ])

        # Collect image file paths
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {category: idx for idx, category in enumerate(categories)}

        for category in categories:
            category_path = os.path.join(root_dir, split, category)
            if os.path.exists(category_path):
                for img_name in os.listdir(category_path):
                    if img_name.endswith(".jpg"):
                        self.image_paths.append(os.path.join(category_path, img_name))
                        self.labels.append(self.class_to_idx[category])
        
        # Ensure we have data
        assert len(self.image_paths) > 0, "No images found in the specified categories."

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply predefined transform
        image = self.transform(image)

        return image, label


# dataset = AFHQDataset(root_dir="/home/shahriar/data/afhq", split="train", categories=["cat", "dog"])
# image, label = dataset[0]
# print(image.shape, label)  # Should output: (3, 256, 256) <label>


class AFHQInMemoryDataset(Dataset):
    def __init__(self, root_dir, split="train", categories=["cat", "dog", "wild"]):
        """
        In-Memory Dataset that loads all images into RAM during initialization.
        """
        assert split in ["train", "val"], "Split must be 'train' or 'val'"

        self.root_dir = root_dir
        self.split = split
        self.categories = categories

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to 256x256
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
        ])

        self.image_paths = []
        # self.labels = []
        self.images = []  # Store images in memory

        self.class_to_idx = {category: idx for idx, category in enumerate(categories)}

        # Load images into memory
        for category in categories:
            category_path = os.path.join(root_dir, split, category)
            if os.path.exists(category_path):
                for img_name in os.listdir(category_path):
                    if img_name.endswith(".jpg"):
                        img_path = os.path.join(category_path, img_name)
                        self.image_paths.append(img_path)
                        # self.labels.append(self.class_to_idx[category])

                        # Load and transform image immediately
                        image = Image.open(img_path).convert("RGB")
                        image = self.transform(image)
                        self.images.append(image)  # Store in memory
        
        # Convert list to tensors for faster indexing
        self.images = torch.stack(self.images)
        # self.labels = torch.tensor(self.labels)
        print(f"Loaded {len(self.images)} images into RAM.")

        self.x0 = torch.randn_like(self.images) #initial dist

    # def push_forward(self, model, nsteps=10, dt=0.1, batch_size=64):
    #     """
    #     Updates self.x0 by applying RK4 steps in mini-batches.
    #     Args:
    #         model: The model used as the derivative function for RK4.
    #         nsteps (int): Number of RK4 steps.
    #         dt (float): Time step for RK4.
    #         batch_size (int): How many images to process at once.
    #     """
    #     # Determine the device on which the model resides (GPU preferred)
    #     device = next(model.parameters()).device
    #     new_x0_batches = []
    #     with torch.no_grad():
    #         # Process self.x0 in chunks of size 'batch_size'
    #         for i in range(0, self.x0.size(0), batch_size):
    #             batch = self.x0[i : i + batch_size].to(device)
    #             # Compute RK4 on this mini-batch
    #             x_updated = RK4(model, batch, dt=dt, nt=nsteps,traj=False)
    #             new_x0_batches.append(x_updated.cpu())
    #     # Concatenate all updated batches to update self.x0
    #     self.x0 = torch.cat(new_x0_batches, dim=0)
    def push_forward(self, model, nsteps=10, dt=0.1, batch_size=64, use_saved=False):
        """
        Updates self.x0 by applying RK4 steps in mini-batches.
        Args:
            model: The model used as the derivative function for RK4.
            nsteps (int): Number of RK4 steps.
            dt (float): Time step for RK4.
            batch_size (int): How many images to process at once.
            use_saved (bool): If True, load x0 from disk if available.
        """
        # Define the file path where x0 is saved/loaded.
        x0_path = os.path.join("/home/shahriar/FlowMatchingPredCor/Cat/stored_tensors", "model0_x0.pt")
    
        # If use_saved is True and the file exists, load x0 and return early.
        if use_saved and os.path.exists(x0_path):
            self.x0 = torch.load(x0_path).cpu()
            print(f"Loaded x0 from {x0_path}")
            return
        # Determine the device on which the model resides (GPU preferred)
        device = next(model.parameters()).device

        # Move the entire tensor to the device once
        with torch.no_grad():
            self.x0 = self.x0.to(device)
            # Process self.x0 in-place in batches
            for i in range(0, self.x0.size(0), batch_size):
                self.x0[i:i + batch_size] = RK4(model, self.x0[i:i + batch_size], dt=dt, nt=nsteps, traj=False)
            # Move the updated tensor back to CPU
            self.x0 = self.x0.cpu()
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Return from preloaded memory
        x0 = self.x0[idx, :]
        idy = torch.randint(0, self.images.shape[0], (1,)).item()
        x1 = self.images[idy, :]
        return x0, x1  


# class AFHQInMemoryDataset_64(Dataset):
#     def __init__(self, root_dir, split="train", categories=["cat", "dog", "wild"]):
#         """
#         In-Memory Dataset that loads all images into RAM during initialization.
#         """
#         assert split in ["train", "val"], "Split must be 'train' or 'val'"

#         self.root_dir = root_dir
#         self.split = split
#         self.categories = categories

#         self.transform = transforms.Compose([
#             transforms.Resize((64, 64)),
#             transforms.ToTensor(),  # Convert to tensor
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1,1]
#             transforms.RandomHorizontalFlip() #March 10 addition
#         ])

#         self.image_paths = []
#         # self.labels = []
#         self.images = []  # Store images in memory

#         self.class_to_idx = {category: idx for idx, category in enumerate(categories)}

#         # Load images into memory
#         for category in categories:
#             category_path = os.path.join(root_dir, split, category)
#             if os.path.exists(category_path):
#                 for img_name in os.listdir(category_path):
#                     if img_name.endswith(".jpg"):
#                         img_path = os.path.join(category_path, img_name)
#                         self.image_paths.append(img_path)
#                         # self.labels.append(self.class_to_idx[category])

#                         # Load and transform image immediately
#                         image = Image.open(img_path).convert("RGB")
#                         image = self.transform(image)
#                         self.images.append(image)  # Store in memory
        
#         # Convert list to tensors for faster indexing
#         self.images = torch.stack(self.images)
#         # self.labels = torch.tensor(self.labels)
#         print(f"Loaded {len(self.images)} images into RAM.")

#         self.x0 = torch.randn_like(self.images) #initial dist


#     def push_forward(self, model, nsteps=10, dt=0.1, batch_size=64, use_saved=False):
#         """
#         Updates self.x0 by applying RK4 steps in mini-batches.
#         Args:
#             model: The model used as the derivative function for RK4.
#             nsteps (int): Number of RK4 steps.
#             dt (float): Time step for RK4.
#             batch_size (int): How many images to process at once.
#             use_saved (bool): If True, load x0 from disk if available.
#         """
#         # Define the file path where x0 is saved/loaded.
#         x0_path = os.path.join("/home/shahriar/FlowMatchingPredCor/Cat/stored_tensors", "model0_x0.pt")
    
#         # If use_saved is True and the file exists, load x0 and return early.
#         if use_saved and os.path.exists(x0_path):
#             self.x0 = torch.load(x0_path).cpu()
#             print(f"Loaded x0 from {x0_path}")
#             return
#         # Determine the device on which the model resides (GPU preferred)
#         device = next(model.parameters()).device

#         # Move the entire tensor to the device once
#         with torch.no_grad():
#             self.x0 = self.x0.to(device)
#             # Process self.x0 in-place in batches
#             for i in range(0, self.x0.size(0), batch_size):
#                 self.x0[i:i + batch_size] = RK4(model, self.x0[i:i + batch_size], dt=dt, nt=nsteps, traj=False)
#             # # Move the updated tensor back to CPU
#             # print("Pushing forward done, adding gaussian noise to new initial state")
#             # self.x0 = (0.9 * self.x0) + (0.1 * torch.randn_like(self.x0))
#             self.x0 = self.x0.cpu()
#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         # Return from preloaded memory
#         x0 = self.x0[idx, :]
#         idy = torch.randint(0, self.images.shape[0], (1,)).item()
#         x1 = self.images[idy, :]
#         return x0, x1 


class AFHQInMemoryDataset_256(Dataset):
    def __init__(self, root_dir, split="train", categories=["cat", "dog", "wild"], use_horizontal_flips=True):
        """
        In-Memory Dataset that loads all images into RAM during initialization.
        Args:
            root_dir: Path to the AFHQ dataset
            split: 'train' or 'val'
            categories: List of categories to include
            use_horizontal_flips: Whether to use random horizontal flips
        """
        assert split in ["train", "test", "train_test_full"], " Split must be 'train', 'test' or 'train_test_full' "
        self.root_dir = root_dir
        self.split = split
        self.categories = categories
        
        # Create transformation pipeline based on whether flips are enabled
        transform_list = [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1,1]
        ]
        
        # Add horizontal flip only if enabled
        if use_horizontal_flips:
            transform_list.append(transforms.RandomHorizontalFlip())
        
        self.transform = transforms.Compose(transform_list)
        
        self.image_paths = []
        self.images = []  # Store images in memory
        self.class_to_idx = {category: idx for idx, category in enumerate(categories)}
        
        # Load images into memory
        for category in categories:
            category_path = os.path.join(root_dir, split, category)
            if os.path.exists(category_path):
                for img_name in os.listdir(category_path):
                    if img_name.endswith(".png"):
                        img_path = os.path.join(category_path, img_name)
                        self.image_paths.append(img_path)
                        
                        # Load and transform image immediately
                        image = Image.open(img_path).convert("RGB")
                        image = self.transform(image)
                        self.images.append(image)  # Store in memory
        
        # Convert list to tensors for faster indexing
        self.images = torch.stack(self.images)
        print(f"Loaded {len(self.images)} images into RAM.")
        self.x0 = torch.randn_like(self.images)  # initial dist

    def push_forward(self, model, nsteps=10, dt=0.1, batch_size=64, use_saved=False):
        """
        Updates self.x0 by applying RK4 steps in mini-batches.
        Args:
            model: The model used as the derivative function for RK4.
            nsteps (int): Number of RK4 steps.
            dt (float): Time step for RK4.
            batch_size (int): How many images to process at once.
            use_saved (bool): If True, load x0 from disk if available.
        """
        # Define the file path where x0 is saved/loaded.
        x0_path = os.path.join("/home/shahriar/FlowMatchingPredCor/Cat/stored_tensors", "model0_x0.pt")
        
        # If use_saved is True and the file exists, load x0 and return early.
        if use_saved and os.path.exists(x0_path):
            self.x0 = torch.load(x0_path).cpu()
            print(f"Loaded x0 from {x0_path}")
            return
        
        # Determine the device on which the model resides (GPU preferred)
        device = next(model.parameters()).device
        
        # Move the entire tensor to the device once
        with torch.no_grad():
            self.x0 = self.x0.to(device)
            
            # Process self.x0 in-place in batches
            for i in range(0, self.x0.size(0), batch_size):
                self.x0[i:i + batch_size] = RK4(model, self.x0[i:i + batch_size], dt=dt, nt=nsteps, traj=False)
            
            self.x0 = self.x0.cpu()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Return from preloaded memory
        x0 = self.x0[idx, :]
        idy = torch.randint(0, self.images.shape[0], (1,)).item()
        x1 = self.images[idy, :]
        return x0, x1


# Add a new PreEncodedDataset class
class PreEncoded_AFHQ_256_Dataset(Dataset):
    """Dataset of pre-encoded latent vectors"""
    def __init__(self, encoded_dataset_path, randomize_initial=True, seed=42):
        """
        Initialize the pre-encoded dataset.
        
        Args:
            encoded_dataset_path: Path to encoded dataset file
            randomize_initial: Whether to use random noise for x0 or fixed noise
            seed: Random seed for reproducibility
        """
        data = torch.load(encoded_dataset_path)
        self.latents = data['latents']  # Shape: [N, C, H, W]
        self.log_vars = data.get('log_vars', None)  # May not exist in older saves
        
        print(f"Loaded {len(self.latents)} pre-encoded samples of shape {self.latents.shape}")
        
        # Initialize x0 (starting point for flow)
        if randomize_initial:
            # Random initialization
            self.x0 = torch.randn_like(self.latents)
        else:
            # Fixed initialization with given seed
            generator = torch.Generator().manual_seed(seed)
            self.x0 = torch.randn_like(self.latents, generator=generator)
            
    def push_forward(self, model, nsteps=10, dt=0.1, batch_size=64, use_saved=False):
        """
        Updates self.x0 by applying RK4 steps in mini-batches.
        Same API as in AFHQInMemoryDataset_256 for compatibility.
        """
        # Define the file path where x0 is saved/loaded.
        x0_path = os.path.join("/home/shahriar/FlowMatchingPredCor/Cat/stored_tensors", "model0_x0.pt")
    
        # If use_saved is True and the file exists, load x0 and return early.
        if use_saved and os.path.exists(x0_path):
            self.x0 = torch.load(x0_path).cpu()
            print(f"Loaded x0 from {x0_path}")
            return
            
        # Determine the device on which the model resides
        device = next(model.parameters()).device

        # Move the entire tensor to the device once
        with torch.no_grad():
            self.x0 = self.x0.to(device)
            # Process self.x0 in-place in batches
            for i in range(0, self.x0.size(0), batch_size):
                self.x0[i:i + batch_size] = RK4(model, self.x0[i:i + batch_size], dt=dt, nt=nsteps, traj=False)
            # Move the updated tensor back to CPU
            self.x0 = self.x0.cpu()

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        # Return random pairs of x0 and latents as in the original dataset
        x0 = self.x0[idx]
        idy = torch.randint(0, self.latents.shape[0], (1,)).item()
        x1 = self.latents[idy]
        return x0, x1
    


class AFHQ_256_EncodedImagePairDataset(Dataset):
    def __init__(self, dataset_path):
        # The original images have already been processed by AFHQInMemoryDataset_256 and have been resized to 256x256 and scaled to [-1,1]
        # Flipped versions also included so the size of this dataset is twice that of AFHQ_256_Cat
        # Load the saved data
        data = torch.load(dataset_path)
        self.latents = data['latents']
        self.original_images = data['original_images']
        self.log_vars = data['log_vars']
        
    def __len__(self):
        return len(self.latents)
        
    def __getitem__(self, idx):
        # Return (encoded_image, original_image) pair
        return self.latents[idx], self.original_images[idx]
    

    ################# CelebA-HQ #######################

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CelebAHQ256Dataset(Dataset):
    def __init__(self, root_dir="/home/shahriar/data/celeba/celeba_hq_256", use_horizontal_flips=False):
        """
        Disk-based dataset for CelebA-HQ 256x256 images.
        Args:
            root_dir (str): Path to the directory with the resized images.
            use_horizontal_flips (bool): If True, apply random horizontal flips.
        """
        self.root_dir = root_dir
        
        # List all PNG files in the root_dir.
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith(".png")
        ]
        
        # Build transformation pipeline.
        transform_list = [
            # The images are already resized to 256x256, but Resize is included for safety.
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        
        # Optionally add horizontal flipping.
        if use_horizontal_flips:
            transform_list.insert(0, transforms.RandomHorizontalFlip())
            
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image from disk.
        image_path = self.image_paths[idx]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
        
        # Apply the transformation pipeline.
        image = self.transform(img)
        
        # Return a tuple with None as the first element and the image as the second.
        return 0.0, image

# # Example usage:
# if __name__ == "__main__":
#     dataset = CelebAHQ256Dataset(use_horizontal_flips=True)
#     print(f"Found {len(dataset)} images.")
#     sample = dataset[0]
#     print("Sample output:", sample[0], sample[1].shape)

import os
import torch
from torch.utils.data import Dataset

class PreEncoded_CelebAHQ256_Dataset(Dataset):
    """
    Pre-encoded CelebA-HQ 256 Dataset.

    This class loads the entire encoded latent tensor (and optional log variance)
    from a .pt file that was created by your encoding script.
    
    For each sample, it returns a tuple (None, latent), following your convention.
    """
    def __init__(self, encoded_path="/home/shahriar/data/celeba/celeba_hq_256_encoded/encoded_dataset.pt"):
        """
        Args:
            encoded_path (str): Path to the .pt file containing the encoded latents.
        """
        data = torch.load(encoded_path)
        self.latents = data['latents']  # Expected shape: [N, C, H, W]
        self.log_vars = data.get('log_vars', None)
        print(f"Loaded {len(self.latents)} pre-encoded samples of shape {self.latents.shape}")

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        latent_sample = self.latents[idx]
        return 0.0, latent_sample

# # Example usage:
# if __name__ == "__main__":
#     dataset = PreEncoded_CelebAHQ256_Dataset(encoded_path="celeba_hq_256_encoded.pt")
#     print(f"Dataset length: {len(dataset)}")
#     sample = dataset[0]
#     print("Sample output (should be (None, latent)):", sample)


########################## LSUN ################################

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob
from tqdm import tqdm


class LSUNDataset(Dataset):
    """
    Base Dataset for LSUN categories (Bedrooms, Churches, etc.)
    """
    def __init__(self, 
                 root_dir, 
                 category, 
                 split="train", 
                 transform=None,
                 use_horizontal_flips=True):
        """
        Args:
            root_dir (str): Root directory of LSUN dataset
            category (str): Category name (e.g., 'bedroom', 'church_outdoor')
            split (str): 'train', 'val', 'test', 'all', or 'fid_train_50k'
            transform (callable, optional): Optional transform to be applied on an image
            use_horizontal_flips (bool): Whether to use random horizontal flips during training
        """
        assert split in ["train", "val", "test", "all", "fid_train_50k"], \
            "Split must be 'train', 'val', 'test', 'all', or 'fid_train_50k'"
        
        self.root_dir = root_dir
        self.category = category
        self.split = split
        
        # Create transformation pipeline
        if transform is None:
            transform_list = [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1,1]
            ]
            
            # Add horizontal flip only if enabled and if training
            if use_horizontal_flips and split in ["train", "all"]:
                transform_list.insert(0, transforms.RandomHorizontalFlip())
            
            self.transform = transforms.Compose(transform_list)
        else:
            self.transform = transform
        
        # Load image paths
        self.image_dir = os.path.join(root_dir, category, split)
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Directory not found: {self.image_dir}")
        
        # Get all image files (both .png and .jpg)
        self.image_paths = sorted(glob.glob(os.path.join(self.image_dir, "*.png")))
        jpg_files = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")))
        self.image_paths.extend(jpg_files)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
        
        print(f"Found {len(self.image_paths):,} images in {self.image_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Return None as first item and image as second item
        return 0.0, image


class LSUNBedroomDataset(LSUNDataset):
    """LSUN Bedroom Dataset"""
    def __init__(self, 
                 root_dir, 
                 split="train", 
                 transform=None,
                 use_horizontal_flips=True):
        
        super().__init__(
            root_dir=root_dir,
            category="bedroom",
            split=split,
            transform=transform,
            use_horizontal_flips=use_horizontal_flips
        )


class LSUNChurchDataset(LSUNDataset):
    """LSUN Church Dataset"""
    def __init__(self, 
                 root_dir, 
                 split="train", 
                 transform=None,
                 use_horizontal_flips=True):
        
        super().__init__(
            root_dir=root_dir,
            category="church_outdoor",
            split=split,
            transform=transform,
            use_horizontal_flips=use_horizontal_flips
        )


# # Example usage:
# if __name__ == "__main__":
#     # Set the path to your processed LSUN dataset
#     lsun_root = "./lsun"
    
#     # Create datasets
#     bedroom_train = LSUNBedroomDataset(root_dir=lsun_root, split="train")
#     church_train = LSUNChurchDataset(root_dir=lsun_root, split="train")
    
#     # For using all data (train+val+test combined)
#     bedroom_all = LSUNBedroomDataset(root_dir=lsun_root, split="all")
    
#     # Create DataLoader for training
#     from torch.utils.data import DataLoader
    
#     train_loader = DataLoader(
#         bedroom_train,
#         batch_size=32,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     # Example iteration
#     for _, images in train_loader:
#         # In flow matching, you might do something like:
#         # loss = flow_matching_loss(model, images)
#         break
    
#     print(f"Loaded batch with shape: {images.shape}")  # Should be [32, 3, 256, 256]


import os
import torch
from torch.utils.data import Dataset

class PreEncoded_LSUNChurch_Dataset(Dataset):
    """
    Pre-encoded LSUN Church Dataset.
    This class loads the entire encoded latent tensor (and optional log variance)
    from a .pt file that was created by your encoding script.
    For each sample, it returns a tuple (None, latent), following your convention.
    """
    def __init__(self, encoded_path="/home/shahriar/data/lsun_encoded/church_train/encoded_dataset.pt"):
        """
        Args:
            encoded_path (str): Path to the .pt file containing the encoded latents.
        """
        data = torch.load(encoded_path)
        self.latents = data['latents']  # Expected shape: [N, C, H, W]
        self.log_vars = data.get('log_vars', None)
        print(f"Loaded {len(self.latents)} pre-encoded LSUN Church samples of shape {self.latents.shape}")
        
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        latent_sample = self.latents[idx]
        return 0.0, latent_sample
    
######################### LSUN BEDROOM #########################
from torch.utils.data import Sampler
import random

# class LSUN_Bedrooms_ChunkAwareBatchSampler(Sampler):
#     def __init__(self, dataset, batch_size, shuffle=True):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.shuffle = shuffle
        
#         # Group indices by chunk
#         self.chunk_indices = {}
#         for i in range(len(dataset)):
#             chunk_idx = i // dataset.chunk_size
#             if chunk_idx not in self.chunk_indices:
#                 self.chunk_indices[chunk_idx] = []
#             self.chunk_indices[chunk_idx].append(i)
    
#     def __iter__(self):
#         # Shuffle the chunks
#         chunks = list(self.chunk_indices.keys())
#         if self.shuffle:
#             chunks = random.sample(chunks, len(chunks))
        
#         # For each chunk, yield batches
#         for chunk in chunks:
#             indices = self.chunk_indices[chunk]
#             if self.shuffle:
#                 indices = random.sample(indices, len(indices))
            
#             # Yield batches from this chunk
#             for i in range(0, len(indices), self.batch_size):
#                 yield indices[i:i + self.batch_size]
    
#     def __len__(self):
#         return (len(self.dataset) + self.batch_size - 1) // self.batch_size

from torch.utils.data import Sampler
import random

class LSUN_Bedrooms_ChunkAwareBatchSampler(Sampler):
    # this is a new one, the commented one above goes with the old dataset class
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset     = dataset
        self.batch_size  = batch_size
        self.shuffle     = shuffle

        chunk_size       = dataset.full_chunk_size
        total            = len(dataset)
        num_chunks       = dataset.num_chunks

        # build ranges, not per-index lists
        self.chunk_ranges = []
        for c in range(num_chunks):
            start = c * chunk_size
            end   = min(start + chunk_size, total)
            self.chunk_ranges.append((start, end))

    def __iter__(self):
        # optionally shuffle chunk order
        chunks = list(range(len(self.chunk_ranges)))
        if self.shuffle:
            random.shuffle(chunks)

        for c in chunks:
            start, end = self.chunk_ranges[c]
            indices    = list(range(start, end))
            if self.shuffle:
                random.shuffle(indices)
            # yield batches of size batch_size
            for i in range(0, len(indices), self.batch_size):
                yield indices[i : i + self.batch_size]

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size



import os
import json
import torch
from torch.utils.data import Dataset
from collections import OrderedDict

class PreEncoded_LSUNBedroom_Dataset(Dataset):
    """
    Memory-efficient loader for chunked LSUN Bedrooms encoded latents.
    Picks up the actual chunk length from the first .pt file to avoid
    any mismatch between metadata['chunk_size'] and how the chunks were written.
    """

    def __init__(self,
                 encoded_dataset_path: str,
                 cache_chunks: int = 1):
        self.encoded_path   = encoded_dataset_path
        self.cache_chunks   = cache_chunks

        # Load metadata.json
        meta_path = os.path.join(self.encoded_path, "metadata.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.total_samples = meta["total_samples"]
        self.num_chunks    = meta["num_chunks"]

        # LRU cache for a small number of chunks in RAM
        self._cache = OrderedDict()

        # --- DISCOVER THE ACTUAL FULL CHUNK SIZE ---
        # Load the very first chunk to see how many latents it actually contains
        first_path = os.path.join(self.encoded_path, "chunk_00000.pt")
        first_data = torch.load(first_path)
        self.full_chunk_size = first_data["latents"].shape[0]
        self.chunk_size= self.full_chunk_size
        # clear any accidental cache entry
        self._cache.clear()

    def __len__(self):
        return self.total_samples

    def _load_chunk(self, chunk_idx: int):
        # LRU‐cache lookup
        if chunk_idx in self._cache:
            self._cache.move_to_end(chunk_idx)
            return self._cache[chunk_idx]

        # otherwise load from disk
        fn   = f"chunk_{chunk_idx:05d}.pt"
        data = torch.load(os.path.join(self.encoded_path, fn))
        self._cache[chunk_idx] = data
        self._cache.move_to_end(chunk_idx)

        # evict oldest if over limit
        if len(self._cache) > self.cache_chunks:
            self._cache.popitem(last=False)

        return data

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range")

        # figure out which chunk (using the *actual* chunk length!)
        chunk_idx = idx // self.full_chunk_size
        chunk_data = self._load_chunk(chunk_idx)

        # recompute the true start of this chunk
        start = chunk_idx * self.full_chunk_size
        local_idx = idx - start

        # now this will never run out of bounds
        x1 = chunk_data["latents"][local_idx]
        x0 = torch.empty(0, dtype=torch.float32)
        return x0, x1


# import os
# import json
# import torch
# from torch.utils.data import Dataset
# from collections import OrderedDict

# class PreEncoded_LSUNBedroom_Dataset(Dataset):
#     """
#     Memory-efficient loader for chunked LSUN Bedrooms encoded latents.
    
#     Each __getitem__ returns (x0, x1) where:
#       - x0 is a placeholder (empty tensor for memory efficiency)
#       - x1 is the latent at the specified index
      
#     Args:
#         encoded_dataset_path (str): Directory with metadata.json and chunk_#####.pt files.
#         cache_chunks (int): How many chunks to keep in RAM at once.
#     """
#     def __init__(self,
#                  encoded_dataset_path: str,
#                  cache_chunks: int = 1):
#         self.encoded_path = encoded_dataset_path
#         self.cache_chunks = cache_chunks
        
#         # Load metadata
#         with open(os.path.join(self.encoded_path, "metadata.json"), "r") as f:
#             meta = json.load(f)
#         self.total_samples = meta["total_samples"]
#         self.chunk_size = meta["chunk_size"]
#         self.num_chunks = meta["num_chunks"]
        
#         # LRU cache for chunks
#         self._cache = OrderedDict()
    
#     def __len__(self):
#         return self.total_samples
    
#     def _load_chunk(self, chunk_idx: int):
#         # Return from cache if available
#         if chunk_idx in self._cache:
#             self._cache.move_to_end(chunk_idx)
#             return self._cache[chunk_idx]
        
#         # Load from disk
#         chunk_file = os.path.join(self.encoded_path, f"chunk_{chunk_idx:05d}.pt")
#         data = torch.load(chunk_file)
#         self._cache[chunk_idx] = data
#         self._cache.move_to_end(chunk_idx)
        
#         # Evict oldest if over limit
#         if len(self._cache) > self.cache_chunks:
#             self._cache.popitem(last=False)
        
#         return data
    
#     def __getitem__(self, idx):
#         # figure out which on-disk chunk to load
#         chunk_idx = idx // self.chunk_size
#         chunk_data = self._load_chunk(chunk_idx)

#         # **instead of** idx % self.chunk_size
#         start = chunk_data['start_idx']       # e.g. 50 * 60000 = 3,000,000
#         local_idx = idx - start               # now guaranteed < len(chunk_data['latents'])

#         x1 = chunk_data['latents'][local_idx]
#         x0 = torch.empty(0, dtype=torch.float32)
#         return x0, x1


############### FFHQ - 256 ######################

import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class FFHQ256Dataset(Dataset):
    """
    PyTorch Dataset for FFHQ 256×256 images stored as PNGs in a single folder.
    Scales pixel values to the range [-1, 1], with optional random horizontal flip.
    """
    def __init__(self, root_dir, flip_horizontal=False):
        """
        Args:
            root_dir (str): Path to the folder containing FFHQ_256 PNG files.
            flip_horizontal (bool): If True, applies random horizontal flip augmentation with p=0.5.
        """
        self.root_dir = root_dir
        self.flip_horizontal = flip_horizontal

        # Gather all PNG image file paths
        self.image_paths = sorted([
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith('.png')
        ])

        # Build transform pipeline: resize → optional flip → to tensor → normalize to [-1,1]
        t_list = [transforms.Resize((256, 256))]
        if self.flip_horizontal:
            t_list.append(transforms.RandomHorizontalFlip(p=0.5))
        t_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        self.transform = transforms.Compose(t_list)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns:
            torch.Tensor: Image tensor scaled to [-1,1], with optional augmentation.
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        return 0.0, self.transform(image)


import os
import torch
from torch.utils.data import Dataset

class PreEncoded_FFHQ_Dataset(Dataset):
    """
    Pre-encoded FFHQ Dataset.
    This class loads the entire encoded latent tensor (and optional log variance)
    from a .pt file that was created by the encoding script.
    For each sample, it returns a tuple (None, latent), following the established convention.
    """
    def __init__(self, encoded_dataset_path="/path/to/ffhq_encoded/encoded_dataset.pt"):
        """
        Args:
            encoded_path (str): Path to the .pt file containing the encoded latents.
        """
        if not os.path.exists(encoded_dataset_path):
            raise FileNotFoundError(f"Encoded dataset file not found at: {encoded_dataset_path}")
            
        data = torch.load(encoded_dataset_path)
        self.latents = data['latents']  # Expected shape: [N, C, H, W]
        self.log_vars = data.get('log_vars', None)
        
        print(f"Loaded {len(self.latents)} pre-encoded FFHQ samples of shape {self.latents.shape}")
        
        # Optional: If you need to check if the dataset was augmented with flips
        if os.path.exists(os.path.dirname(encoded_dataset_path) + "/metadata.txt"):
            with open(os.path.dirname(encoded_dataset_path) + "/metadata.txt", "r") as f:
                metadata = f.read()
                if "Using flipped augmentation: True" in metadata:
                    print("This dataset includes horizontal flip augmentation")
    
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        latent_sample = self.latents[idx]
        
        # If log_vars are available and you want to return them as well:
        # log_var_sample = self.log_vars[idx] if self.log_vars is not None else None
        # return 0.0, latent_sample, log_var_sample
        
        # Following the same convention as your other datasets:
        return 0.0, latent_sample