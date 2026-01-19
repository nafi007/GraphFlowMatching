import os
import torch
from torch.utils.data import DataLoader, Dataset
import argparse
from tqdm import tqdm
import numpy as np
from torchvision import transforms

# Import your existing components
from AFHQ_256 import AFHQInMemoryDataset_256
from networks import create_hf_vae_wrappers

class AugmentedDataset(Dataset):
    """Dataset that creates a flipped counterpart for each image"""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.length = len(base_dataset)
        self.flip_transform = transforms.RandomHorizontalFlip(p=1.0)  # Always flip
        
        # Create a random noise tensor similar in shape to the dataset items
        # This is just for the first return value that we don't use
        _, sample_img = self.base_dataset[0]
        self.dummy_x0 = torch.randn_like(sample_img)
        
    def __len__(self):
        return self.length * 2  # Double the size
        
    def __getitem__(self, idx):
        is_flipped = idx >= self.length
        original_idx = idx % self.length
        
        # Get original image - x0 is just a dummy placeholder
        _, img = self.base_dataset[original_idx]
        
        if is_flipped:
            # Create a flipped version
            flipped_img = self.flip_transform(img)
            return self.dummy_x0, flipped_img
        else:
            # Return the original
            return self.dummy_x0, img

def create_encoded_dataset(args):
    """
    Create a dataset of encoded latent vectors from AFHQ images.
    With option to augment by creating flipped versions of all images.
    
    Args:
        args: Command-line arguments
    """
    # Create the output directory if it doesn't exist
    os.makedirs(args.encoded_dataset_path, exist_ok=True)
    
    # Load the dataset
    print(f"Loading dataset from {args.datapath}")
    base_dataset = AFHQInMemoryDataset_256(
        root_dir=args.datapath, 
        split="train", 
        categories=["cat"], 
        use_horizontal_flips=False  # No random augmentation, we'll do it deterministically
    )
    
    # Apply augmentation if requested
    if args.use_flipped_augmentation:
        print("Creating augmented dataset with flipped counterparts...")
        dataset = AugmentedDataset(base_dataset)
        print(f"Original dataset size: {len(base_dataset)}, Augmented size: {len(dataset)}")
    else:
        dataset = base_dataset
    
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # No need to shuffle for encoding
        num_workers=args.num_workers
    )
    
    # Load or create encoder
    print(f"Loading VAE from {args.hf_vae_model}")
    encoder, _ = create_hf_vae_wrappers(
        pretrained_model_name_or_path=args.hf_vae_model,
        device=args.device
    )
    encoder.eval()
    
    # Process batches and encode
    encoded_latents = []
    log_vars = []  # Store log variance for potential use
    
    print("Encoding images...")
    with torch.no_grad():
        for batch_idx, (_, x1) in enumerate(tqdm(train_loader)):
            x1 = x1.to(args.device)
            latents, log_var = encoder(x1)
            
            # Store the latents and log_vars
            encoded_latents.append(latents.cpu())
            log_vars.append(log_var.cpu())
            
    # Concatenate all batches
    all_latents = torch.cat(encoded_latents, dim=0)
    all_log_vars = torch.cat(log_vars, dim=0)
    
    print(f"Encoded {len(all_latents)} images to latent vectors of shape {all_latents.shape}")
    
    # Save the encoded latents
    torch.save(
        {
            'latents': all_latents,
            'log_vars': all_log_vars,
        }, 
        os.path.join(args.encoded_dataset_path, "encoded_dataset.pt")
    )
    
    # Also save metadata about the encoding
    with open(os.path.join(args.encoded_dataset_path, "metadata.txt"), "w") as f:
        f.write(f"Original dataset: {args.datapath}\n")
        f.write(f"VAE model used: {args.hf_vae_model}\n")
        f.write(f"Number of encoded samples: {len(all_latents)}\n")
        f.write(f"Latent shape: {all_latents.shape}\n")
        f.write(f"Using flipped augmentation: {args.use_flipped_augmentation}\n")
        if args.use_flipped_augmentation:
            f.write(f"Original dataset size: {len(base_dataset)}, Augmented size: {len(dataset)}\n")
    
    print(f"Encoded dataset saved to {args.encoded_dataset_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create encoded dataset from AFHQ images")
    
    parser.add_argument('--datapath', type=str, default="/data/shahriar/datasets/afhq_v2", 
                        help='Path to AFHQ dataset')
    parser.add_argument('--encoded_dataset_path', type=str, default="/data/shahriar/datasets/afhq_v2_encoded",
                        help='Path to save encoded dataset')
    parser.add_argument('--hf_vae_model', type=str, default="stabilityai/sd-vae-ft-ema",
                        help='Hugging Face VAE model name or path')
    parser.add_argument('--device', type=str, default='cuda:1',
                        help='Device to use for encoding')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for encoding')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for data loading')
    
    # Add new flag for augmentation
    parser.add_argument('--use_flipped_augmentation', action='store_true',
                        help='Augment dataset with flipped counterparts of all images')
    parser.set_defaults(use_flipped_augmentation=True)
    
    args = parser.parse_args()
    create_encoded_dataset(args)