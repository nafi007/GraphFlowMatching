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
                 use_horizontal_flips=True,
                 return_noise=True,
                 integrator=None,  # Function for numerical integration
                 cached_noise_path=None):
        """
        Args:
            root_dir (str): Root directory of LSUN dataset
            category (str): Category name (e.g., 'bedroom', 'church_outdoor')
            split (str): 'train', 'val', or 'fid_train_50k'
            transform (callable, optional): Optional transform to be applied on an image
            use_horizontal_flips (bool): Whether to use random horizontal flips
            return_noise (bool): Whether to return noise tensor with the image
            integrator (callable, optional): Function for numerical integration (RK4, Euler, etc.)
            cached_noise_path (str, optional): Path to save/load cached noise tensors
        """
        assert split in ["train", "val", "fid_train_50k"], "Split must be 'train', 'val', or 'fid_train_50k'"
        
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.return_noise = return_noise
        self.integrator = integrator
        self.cached_noise_path = cached_noise_path
        
        # Create transformation pipeline
        if transform is None:
            transform_list = [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize to [-1,1]
            ]
            
            # Add horizontal flip only if enabled and if training
            if use_horizontal_flips and split == "train":
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
        
        print(f"Found {len(self.image_paths)} images in {self.image_dir}")
        
        # Initialize noise tensor
        if self.return_noise:
            self._initialize_noise()
    
    def _initialize_noise(self):
        """Initialize or load the noise tensor."""
        # Create a filename for the cached noise
        if self.cached_noise_path:
            os.makedirs(os.path.dirname(self.cached_noise_path), exist_ok=True)
            noise_file = self.cached_noise_path
        else:
            # Default location if not specified
            cache_dir = os.path.join(self.root_dir, "noise_cache")
            os.makedirs(cache_dir, exist_ok=True)
            noise_file = os.path.join(cache_dir, f"{self.category}_{self.split}_noise.pt")
        
        # Try to load cached noise
        if os.path.exists(noise_file):
            try:
                self.noise = torch.load(noise_file)
                print(f"Loaded cached noise from {noise_file}")
                if len(self.noise) != len(self.image_paths):
                    print(f"Warning: Cached noise size ({len(self.noise)}) doesn't match dataset size ({len(self.image_paths)})")
                    print("Regenerating noise tensors...")
                    self.noise = torch.randn(len(self.image_paths), 3, 256, 256)
                    torch.save(self.noise, noise_file)
                return
            except Exception as e:
                print(f"Error loading cached noise: {e}")
        
        # Generate new noise
        print("Generating new noise tensors...")
        self.noise = torch.randn(len(self.image_paths), 3, 256, 256)
        
        # Save for future use
        torch.save(self.noise, noise_file)
        print(f"Saved noise tensors to {noise_file}")
    
    def push_forward(self, model, nsteps=10, dt=0.1, batch_size=64, device=None, save_path=None):
        """
        Updates noise tensors by applying numerical integration in mini-batches.
        
        Args:
            model: The model used as the derivative function for integration.
            nsteps (int): Number of integration steps.
            dt (float): Time step for integration.
            batch_size (int): How many tensors to process at once.
            device: Device to use for computation (if None, auto-detect).
            save_path (str, optional): Where to save the updated noise.
        """
        if not self.return_noise:
            print("Warning: push_forward called but return_noise is False")
            return
            
        if self.integrator is None:
            raise ValueError("No integrator function provided. Set integrator in the constructor.")
        
        # Determine device
        if device is None:
            device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Define save path
        if save_path is None and self.cached_noise_path:
            save_path = self.cached_noise_path
        elif save_path is None:
            cache_dir = os.path.join(self.root_dir, "noise_cache")
            os.makedirs(cache_dir, exist_ok=True)
            save_path = os.path.join(cache_dir, f"{self.category}_{self.split}_pushed_noise.pt")
        
        # If the pushed noise already exists, load it
        if os.path.exists(save_path):
            try:
                self.noise = torch.load(save_path)
                print(f"Loaded existing pushed noise from {save_path}")
                return
            except Exception as e:
                print(f"Error loading pushed noise: {e}")
        
        # Process noise in batches
        with torch.no_grad():
            for i in tqdm(range(0, len(self.noise), batch_size), desc="Pushing noise forward"):
                batch = self.noise[i:i + batch_size].to(device)
                
                # Apply the integrator to the batch
                pushed_batch = self.integrator(model, batch, dt=dt, nt=nsteps)
                
                # Update the noise tensor
                self.noise[i:i + batch_size] = pushed_batch.cpu()
        
        # Save the updated noise
        torch.save(self.noise, save_path)
        print(f"Pushed noise saved to {save_path}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Return based on return_noise flag
        if self.return_noise:
            return self.noise[idx], image
        else:
            return None, image


class LSUNBedroomDataset(LSUNDataset):
    """LSUN Bedroom Dataset"""
    def __init__(self, 
                 root_dir, 
                 split="train", 
                 transform=None,
                 use_horizontal_flips=True,
                 return_noise=True,
                 integrator=None,
                 cached_noise_path=None):
        
        super().__init__(
            root_dir=root_dir,
            category="bedroom",
            split=split,
            transform=transform,
            use_horizontal_flips=use_horizontal_flips,
            return_noise=return_noise,
            integrator=integrator,
            cached_noise_path=cached_noise_path
        )


class LSUNChurchDataset(LSUNDataset):
    """LSUN Church Dataset"""
    def __init__(self, 
                 root_dir, 
                 split="train", 
                 transform=None,
                 use_horizontal_flips=True,
                 return_noise=True,
                 integrator=None,
                 cached_noise_path=None):
        
        super().__init__(
            root_dir=root_dir,
            category="church_outdoor",
            split=split,
            transform=transform,
            use_horizontal_flips=use_horizontal_flips,
            return_noise=return_noise,
            integrator=integrator, 
            cached_noise_path=cached_noise_path
        )


# Example integrator functions

def RK4(model, x, dt=0.1, nt=10, traj=False):
    """
    Fourth-order Runge-Kutta method for integration.
    
    Args:
        model: Model that computes the vector field (time derivative)
        x: Input tensor
        dt: Time step
        nt: Number of steps
        traj: Whether to return the full trajectory
    
    Returns:
        The result after integration or the full trajectory
    """
    if traj:
        trajectory = [x.clone()]
    
    for _ in range(nt):
        k1 = model(x)
        k2 = model(x + dt * k1 / 2.0)
        k3 = model(x + dt * k2 / 2.0)
        k4 = model(x + dt * k3)
        
        x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        
        if traj:
            trajectory.append(x.clone())
    
    if traj:
        return torch.stack(trajectory)
    else:
        return x


def Euler(model, x, dt=0.1, nt=10, traj=False):
    """
    Euler method for integration.
    
    Args:
        model: Model that computes the vector field (time derivative)
        x: Input tensor
        dt: Time step
        nt: Number of steps
        traj: Whether to return the full trajectory
    
    Returns:
        The result after integration or the full trajectory
    """
    if traj:
        trajectory = [x.clone()]
    
    for _ in range(nt):
        x = x + dt * model(x)
        
        if traj:
            trajectory.append(x.clone())
    
    if traj:
        return torch.stack(trajectory)
    else:
        return x


# Example usage:

# Define a simple model for testing
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 3, 3, padding=1)
        )
        
    def forward(self, x):
        return self.net(x)

# Usage example:
if __name__ == "__main__":
    # Example for testing
    # LSUN root directory
    lsun_root = "/home/shahriar/data/lsun"
    
    # Create dataset with RK4 integrator
    bedroom_dataset = LSUNBedroomDataset(
        root_dir=lsun_root,
        split="fid_train_50k",  # Use the smaller subset for testing
        use_horizontal_flips=True,
        return_noise=True,
        integrator=RK4,
        cached_noise_path="/home/shahriar/FlowMatchingPredCor/Cat256/cached_noise/bedroom_noise.pt"
    )
    
    # Create a simple model
    model = SimpleModel().cuda()
    
    # Push forward (this updates the noise tensors)
    bedroom_dataset.push_forward(model, nsteps=5, dt=0.1, batch_size=32)
    
    # Get a sample
    noise, image = bedroom_dataset[0]
    print(f"Noise shape: {noise.shape}, Image shape: {image.shape}")