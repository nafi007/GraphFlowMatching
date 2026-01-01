import os
import sys
# Add the parent directory of folder1 to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from networks.networks import create_hf_vae_wrappers
# Add this to your imports at the top of the file
from torchvision.utils import save_image, make_grid

import torch
import numpy as np
import random



import os
import torch
from torch.utils.data import DataLoader

# Assuming the following functions/datasets are defined/imported elsewhere:
# - create_hf_vae_wrappers(pretrained_model_name_or_path, device)
# - PreEncoded_CelebAHQ256_Dataset, PreEncoded_AFHQ_256_Dataset
# - LSUNBedroomDataset, CelebAHQ256Dataset, AFHQInMemoryDataset_256
# - dataloader_kwargs

def ensure_vae_wrappers(model_dir, hf_vae_model, device):
    """
    Ensure that the Hugging Face VAE wrappers exist in the given model directory.
    If they do not exist, create and save them.

    Args:
        model_dir (str): Directory where VAE wrappers are saved.
        hf_vae_model (str): Hugging Face model name or path.
        device (torch.device): Device to load the model on.

    Returns:
        encoder, decoder: The HF VAE encoder and decoder wrappers.
    """
    encoder_path = os.path.join(model_dir, "hf_encoder_wrapper.pt")
    decoder_path = os.path.join(model_dir, "hf_decoder_wrapper.pt")

    need_encoder = not os.path.exists(encoder_path)
    need_decoder = not os.path.exists(decoder_path)

    if need_encoder or need_decoder:
        print(f"Creating missing VAE wrappers using {hf_vae_model}...")
        encoder_wrapper, decoder_wrapper = create_hf_vae_wrappers(
            pretrained_model_name_or_path=hf_vae_model, device=device
        )
        if need_encoder:
            print(f"Saving encoder wrapper to {encoder_path}")
            torch.save(encoder_wrapper, encoder_path)
        if need_decoder:
            print(f"Saving decoder wrapper to {decoder_path}")
            torch.save(decoder_wrapper, decoder_path)

    encoder = torch.load(encoder_path, map_location=device, weights_only=False)
    decoder = torch.load(decoder_path, map_location=device, weights_only=False)

    encoder.eval()
    decoder.eval()
    return encoder, decoder


def load_vae(args, load_encoder=True, load_decoder=True):
    """
    Load either a custom VAE or a Hugging Face VAE based on command-line arguments.

    Args:
        args: Command-line arguments.
        load_encoder (bool): Whether to load the encoder.
        load_decoder (bool): Whether to load the decoder.

    Returns:
        encoder, decoder: The loaded encoder and decoder models.
    """
    if args.use_hf_vae:
        print(f"Using Hugging Face VAE: {args.hf_vae_model}")
        enc, dec = ensure_vae_wrappers(args.model_savepath, args.hf_vae_model, args.device)
    else:
        print("Using custom VAE models")
        try:
            if load_encoder:
                print("Loading custom VAE encoder")
                enc = torch.load(os.path.join(args.model_savepath, 'encoder.pt'),
                                 map_location=args.device, weights_only=False)
            if load_decoder:
                print("Loading custom VAE decoder")
                dec = torch.load(os.path.join(args.model_savepath, 'decoder.pt'),
                                 map_location=args.device, weights_only=False)
        except FileNotFoundError:
            raise ValueError("Custom VAE models not found. Please train a custom VAE first or use --use_hf_vae")
    return enc, dec


from datasets.datasets_256 import AFHQInMemoryDataset_256, PreEncoded_AFHQ_256_Dataset, CelebAHQ256Dataset, PreEncoded_CelebAHQ256_Dataset, PreEncoded_FFHQ_Dataset, FFHQ256Dataset
# from datasets.lsun_datasets import LSUNBedroomDataset
from datasets.datasets_256 import LSUNChurchDataset, LSUNBedroomDataset, PreEncoded_LSUNChurch_Dataset, PreEncoded_LSUNBedroom_Dataset, LSUN_Bedrooms_ChunkAwareBatchSampler
def setup_training_components(args, dataloader_kwargs=None, decoder_require_gradients=False):
    """
    Set up the components needed for training based on whether we're using
    pre-encoded data or not.

    Args:
        args: Command-line arguments

    Returns:
        train_loader: DataLoader for training
        enc: VAE encoder (or None if using pre-encoded data)
        dec: VAE decoder
    """
    if dataloader_kwargs is None:
        dataloader_kwargs = {}
    if args.use_pre_encoded:
        # Set up pre-encoded dataset
        if args.dataset == 'celeba-hq':
            train_dataset = PreEncoded_CelebAHQ256_Dataset(
                encoded_path=os.path.join(args.encoded_dataset_path, "encoded_dataset.pt")
            )
        elif args.dataset == 'lsun_church':
            train_dataset = PreEncoded_LSUNChurch_Dataset(encoded_path=os.path.join(args.encoded_dataset_path, "encoded_dataset.pt"))
        elif args.dataset == 'lsun_bedrooms':
            train_dataset = PreEncoded_LSUNBedroom_Dataset(
                encoded_dataset_path=args.encoded_dataset_path, 
                cache_chunks=args.lsun_bedrooms_dataset_cache_chunks)
            args.num_workers = 0  # massive memory hit if you have more than a few, so just use 0 workers
            batch_sampler = LSUN_Bedrooms_ChunkAwareBatchSampler(
                train_dataset, 
                batch_size=args.train_batch_size, 
                shuffle=True)
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                num_workers=args.num_workers,
                pin_memory=True,
                **dataloader_kwargs)
        elif args.dataset == 'ffhq':
            train_dataset = PreEncoded_FFHQ_Dataset(
                encoded_dataset_path=os.path.join(args.encoded_dataset_path, "encoded_dataset.pt")
            )
        elif args.dataset == 'imnet':
            from datasets.imagenet_dataset import make_imagenet_train_loader, ImageNetLatents
            print(f"Loading ImageNet ILSVRC dataset. Using pre-encoded ImageNet dataset from {args.encoded_dataset_path}")
            # train_dataset, train_loader = make_imagenet_train_loader(root="/data/shahriar/datasets/kagglehub_cache/datasets/ImageNet_ILSVRC_TrainSet/thbdh5765/ilsvrc2012/versions/1", image_size=256, batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, persistent_workers=True)
            train_dataset = ImageNetLatents(
                root=args.encoded_dataset_path,
                device=args.device,
                dtype="fp32",
                strict=True
            )
        else:
            print(f"Using pre-encoded dataset from {args.encoded_dataset_path}")
            train_dataset = PreEncoded_AFHQ_256_Dataset(
                encoded_dataset_path=os.path.join(args.encoded_dataset_path, "encoded_dataset.pt"),
                randomize_initial=True
            )

        if args.dataset not in ['lsun_bedrooms']:
            if args.subset_frac < 1.0:
                total_samples = len(train_dataset)
                subset_size = int(total_samples * args.subset_frac)
                import numpy as np
                indices = np.random.permutation(total_samples)[:subset_size].tolist()
                from torch.utils.data import Subset
                subset_dataset = Subset(train_dataset, indices)
                # Preserve chunk_size for LSUN bedroom sampler compatibility
                if hasattr(train_dataset, 'chunk_size'):
                    subset_dataset.chunk_size = train_dataset.chunk_size
                train_dataset = subset_dataset
                print(f"Using subset of {subset_size}/{total_samples} samples ({args.subset_frac*100:.1f}%) for training.") 
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.train_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                **dataloader_kwargs
            )

        # Only the decoder is needed for generation in pre-encoded mode.
        if args.use_hf_vae:
            try:
                dec = torch.load(os.path.join(args.model_savepath, 'hf_decoder_wrapper.pt'),
                                 map_location=args.device, weights_only=False)
                print("Loaded saved HF VAE decoder wrapper")
            except FileNotFoundError:
                print("Creating new HF VAE decoder wrapper")
                # Create both wrappers but only use the decoder.

                if decoder_require_gradients==False:
                    _, dec = create_hf_vae_wrappers(
                        pretrained_model_name_or_path=args.hf_vae_model,
                        device=args.device, decoder_require_gradients=False)
                else:
                    _, dec = create_hf_vae_wrappers(
                        pretrained_model_name_or_path=args.hf_vae_model,
                        device=args.device,
                        decoder_require_gradients=True)
                    
                os.makedirs(args.model_savepath, exist_ok=True)
                torch.save(dec, os.path.join(args.model_savepath, "hf_decoder_wrapper.pt"))
        else:
            dec = torch.load(os.path.join(args.model_savepath, 'decoder.pt'),
                             map_location=args.device, weights_only=False)
        enc = None

    else:
        # Set up dataset for non-pre-encoded data
        if args.dataset == 'lsun_bedrooms':
            print(f"Using original LSUN BedRoom dataset from {args.datapath}")
            train_dataset = LSUNBedroomDataset(
                root_dir="/home/shahriar/data/lsun",
                split="train",  # or "val" or "fid_train_50k"
                return_noise=False,
                integrator=None,  # e.g. RK4
                cached_noise_path=None,
                use_horizontal_flips=False
            )
        elif args.dataset == 'celeba-hq':
            train_dataset = CelebAHQ256Dataset(root_dir=args.datapath, split="train" , use_horizontal_flips=False)
        elif args.dataset == 'lsun_church':
            train_dataset = LSUNChurchDataset(root_dir=args.datapath, use_horizontal_flips=False)
        elif args.dataset == 'lsun_bedrooms':
            train_dataset = LSUNBedroomDataset(root_dir=args.datapath, use_horizontal_flips=False)
        elif args.dataset == 'ffhq':
            train_dataset = FFHQ256Dataset(root_dir=args.datapath, use_horizontal_flips=False)
        else:
            print(f"Using original AFHQ Cat dataset from {args.datapath}")
            train_dataset = AFHQInMemoryDataset_256(
                root_dir=args.datapath,
                split="train_test_full",
                categories=["cat"],
                use_horizontal_flips=True
            )

        if args.subset_frac < 1.0:
            total_samples = len(train_dataset)
            subset_size = int(total_samples * args.subset_frac)
            import numpy as np
            indices = np.random.permutation(total_samples)[:subset_size].tolist()
            from torch.utils.data import Subset
            subset_dataset = Subset(train_dataset, indices)
            # Preserve chunk_size for LSUN bedroom sampler compatibility
            if hasattr(train_dataset, 'chunk_size'):
                subset_dataset.chunk_size = train_dataset.chunk_size
            train_dataset = subset_dataset
            print(f"Using subset of {subset_size}/{total_samples} samples ({args.subset_frac*100:.1f}%) for training.")
            
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            **dataloader_kwargs)

        # Load or create the VAE (either custom or HF-based)
        enc, dec = load_vae(args)

    return train_loader, enc, dec

def load_pretrained_unet_weights(model, checkpoint_path, device="cuda"):
    """
    Loads pretrained weights from a checkpoint file into an existing UNet model.

    Parameters:
    -----------
    model : torch.nn.Module
        The UNet model instance (already constructed, e.g. via load_adm_unet).
    checkpoint_path : str
        Path to the checkpoint file (.pth) containing the trained weights.
    device : str (default: "cuda")
        The device to map the weights to.

    Returns:
    --------
    model : torch.nn.Module
        The UNet model with loaded weights, set to evaluation mode.
    """
    import torch
    from collections import OrderedDict

    # Load the checkpoint from file
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract the state dict (handle various checkpoint formats)
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_dict' in checkpoint:            # ← handle your content.pth
            state_dict = checkpoint['model_dict']
        else:
            state_dict = checkpoint


    # Remove "module." prefix if the state dict was saved from DataParallel
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v

    # Load the state dict into the model (non-strict to allow minor mismatches)
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    if missing_keys:
        print("Warning: Missing keys in state_dict:", missing_keys)
    if unexpected_keys:
        print("Warning: Unexpected keys in state_dict:", unexpected_keys)
    
    # Set the model to evaluation mode
    model.eval()
    return model



def load_pretrained_dit_model(model_path, model_type="DiT-L/2", img_resolution=32, 
                             in_channels=4, device="cuda", num_classes=1, label_dropout=0.0):
    import torch
    from networks.DiT import DiT_models
    
    # Initialize with the same parameters you've been using
    model_fn = DiT_models[model_type]
    model = model_fn(
        img_resolution=img_resolution,
        in_channels=in_channels,
        num_classes=num_classes,
        label_dropout=label_dropout,
        learn_sigma=False
    ).to(device)
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract and process state dict
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Handle DataParallel prefix
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    
    # Load state dict
    model.load_state_dict(new_state_dict)
    model.eval()
    
    return model

def set_seed(seed=None):
    """
    Set seed for reproducibility across all libraries.
    If seed is None, randomness will not be controlled.
    
    Args:
        seed: Integer seed for reproducibility, or None for random behavior
        
    Returns:
        seed_worker: Worker init function for DataLoader
        cpu_generator: PyTorch CPU generator for DataLoader
        cuda_generator: PyTorch CUDA generator for tensor operations (or None)
    """
    if seed is None:
        # Return None for all to indicate we want randomness
        return None, None, None
    
    # Set all seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For DataLoader workers
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    # Create a CPU generator for DataLoader
    cpu_generator = torch.Generator()
    cpu_generator.manual_seed(seed)
    
    # Create a CUDA generator for tensor operations
    cuda_generator = torch.Generator(device='cuda')
    cuda_generator.manual_seed(seed)
    
    print(f"Set random seed to: {seed}")
    return seed_worker, cpu_generator, cuda_generator

####### IMAGE GENERATION #######

# def integrate_ode(model, x, dt, nt, method='rk4', base_model='resnet', traj=False, rtol=1e-3, atol=1e-6, time_convention='standard'):
#     """
#     Integrate ODE using the specified numerical method
    
#     Parameters:
#     -----------
#     model : torch.nn.Module
#         The model for computing vector field
#     x : torch.Tensor
#         Initial state
#     dt : float
#         Time step size (absolute value)
#     nt : int
#         Number of time steps
#     method : str
#         Integration method ('rk4', 'rk2', 'euler', 'dopri5')
#     base_model : str
#         Model type ('dit' or other)
#     traj : bool
#         Whether to return the full trajectory
#     rtol : float
#         Relative tolerance (for adaptive methods)
#     atol : float
#         Absolute tolerance (for adaptive methods)
#     time_convention : str
#         'standard': t=0 is noise, t=1 is data (reversed convention)   
#         'reverse': t=1 is noise, t=0 is data (usual Flow Matching convention)
    
#     Returns:
#     --------
#     x_final : torch.Tensor
#         Final state after integration.
#     trajectory : torch.Tensor (if traj is True)
#         The full trajectory of states.
#     model_calls : int
#         The total number of function evaluations (calls to the model).
#     """
#     # Set initial and target times based on convention
#     if time_convention == 'reverse':
#         # Reverse convention (official): t=1 is noise, t=0 is data
#         t_start = torch.ones(x.shape[0], device=x.device)
#         t_end = 0.0
#         actual_dt = -abs(dt)  # Negative step to move toward 0
#     else:
#         # Standard convention: t=0 is noise, t=1 is data
#         t_start = torch.zeros(x.shape[0], device=x.device)
#         t_end = 1.0
#         actual_dt = abs(dt)  # Positive step to move toward 1
    
#     t = t_start.clone()
#     X = [x] if traj else None
#     current_x = x

#     # Counter for the number of function evaluations
#     model_calls = 0
    
#     with torch.no_grad():
#         if method == 'dopri5':
#             # Dormand-Prince coefficients
#             a = [
#                 [0],
#                 [1/5],
#                 [3/40, 9/40],
#                 [44/45, -56/15, 32/9],
#                 [19372/6561, -25360/2187, 64448/6561, -212/729],
#                 [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
#             ]
            
#             b1 = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]  # 5th order
#             b2 = [5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40]  # 4th order
            
#             # Function to compute single step with error estimate
#             def dopri5_step(x, t, h, model, is_dit):
#                 nonlocal model_calls
#                 k = []
                
#                 # Compute k1
#                 if is_dit:
#                     k1 = model(x, t)[0].reshape(x.shape)
#                     model_calls += 1
#                 else:
#                     k1 = model(x, t)[0].reshape(x.shape)
#                     model_calls += 1
#                 k.append(k1)
                
#                 # Compute k2 through k6
#                 for i in range(1, 6):
#                     x_new = x.clone()
#                     for j in range(i):
#                         x_new = x_new + h * a[i][j] * k[j]
                    
#                     t_new = t + h * sum(a[i][:i])
                    
#                     if is_dit:
#                         ki = model(x_new, t_new)[0].reshape(x.shape)
#                         model_calls += 1
#                     else:
#                         ki = model(x_new, t_new)[0].reshape(x.shape)
#                         model_calls += 1
#                     k.append(ki)
                
#                 # Compute k7 (for error estimate only)
#                 x_new = x.clone()
#                 for j in range(6):
#                     x_new = x_new + h * b1[j] * k[j]
                
#                 t_new = t + h
                
#                 if is_dit:
#                     k7 = model(x_new, t_new)[0].reshape(x.shape)
#                     model_calls += 1
#                 else:
#                     k7 = model(x_new, t_new)[0].reshape(x.shape)
#                     model_calls += 1
#                 k.append(k7)
                
#                 # Compute the solutions using 5th and 4th order methods
#                 x_5th = x.clone()
#                 for j in range(6):
#                     x_5th = x_5th + h * b1[j] * k[j]
                
#                 x_4th = x.clone()
#                 for j in range(7):
#                     x_4th = x_4th + h * b2[j] * k[j]
                
#                 # Error estimate
#                 error = torch.abs(x_5th - x_4th)
#                 error_ratio = torch.max(error / (atol + rtol * torch.abs(x_5th)))
                
#                 return x_5th, error_ratio
            
#             # Adaptive step size integration
#             is_dit = base_model == 'dit'
#             current_h = actual_dt  # Start with specified step size
#             current_t = t_start[0].item()
            
#             # Define integration direction check
#             if time_convention == 'reverse':
#                 # Going from 1 -> 0
#                 not_done = lambda c_t: c_t > t_end
#                 will_overshoot = lambda c_t, c_h: c_t + c_h < t_end
#             else:
#                 # Going from 0 -> 1
#                 not_done = lambda c_t: c_t < t_end
#                 will_overshoot = lambda c_t, c_h: c_t + c_h > t_end
            
#             while not_done(current_t):
#                 # Adjust step size to not overshoot target
#                 if will_overshoot(current_t, current_h):
#                     current_h = t_end - current_t
                
#                 # Try a step
#                 x_new, error_ratio = dopri5_step(current_x, t, current_h, model, is_dit)
                
#                 # Adjust step size based on error
#                 if error_ratio <= 1.0:
#                     # Step accepted
#                     current_x = x_new
#                     current_t += current_h
#                     t += current_h
                    
#                     if traj:
#                         X.append(current_x)
                    
#                     # Adjust step size based on error and direction
#                     if error_ratio < 0.1:
#                         new_h = current_h * 2.0
#                         # Make sure we don't overshoot in the next step
#                         if time_convention == 'reverse':
#                             current_h = max(new_h, t_end - current_t)
#                         else:
#                             current_h = min(new_h, t_end - current_t)
#                 else:
#                     # Step rejected, reduce step size
#                     current_h = current_h * 0.5
            
#             # In dopri5 branch, x is updated in current_x
#             x = current_x
            
#         else:
#             # Original methods (RK4, RK2, Euler)
#             for i in range(nt):
#                 # Determine model order (dit takes (t,x), others take (x,t))
#                 is_dit = base_model == 'dit'
                
#                 if method == 'rk4':
#                     # RK4 integration
#                     if is_dit:
#                         k1 = model(x, t)[0].reshape(x.shape)
#                         model_calls += 1
#                         k2 = model(x + actual_dt/2 * k1, t + actual_dt/2)[0].reshape(x.shape)
#                         model_calls += 1
#                         k3 = model(x + actual_dt/2 * k2, t + actual_dt/2)[0].reshape(x.shape)
#                         model_calls += 1
#                         k4 = model(x + actual_dt * k3, t + actual_dt)[0].reshape(x.shape)
#                         model_calls += 1
#                     else:
#                         k1 = model(x, t)[0].reshape(x.shape)
#                         model_calls += 1
#                         k2 = model(x + actual_dt/2 * k1, t + actual_dt/2)[0].reshape(x.shape)
#                         model_calls += 1
#                         k3 = model(x + actual_dt/2 * k2, t + actual_dt/2)[0].reshape(x.shape)
#                         model_calls += 1
#                         k4 = model(x + actual_dt * k3, t + actual_dt)[0].reshape(x.shape)
#                         model_calls += 1
#                     step = 1/6 * (k1 + 2*k2 + 2*k3 + k4)
                
#                 elif method == 'rk2':
#                     # RK2 integration
#                     if is_dit:
#                         k1 = model(x, t)[0].reshape(x.shape)
#                         model_calls += 1
#                         k2 = model(x + actual_dt * k1, t + actual_dt)[0].reshape(x.shape)
#                         model_calls += 1
#                     else:
#                         k1 = model(x, t)[0].reshape(x.shape)
#                         model_calls += 1
#                         k2 = model(x + actual_dt * k1, t + actual_dt)[0].reshape(x.shape)
#                         model_calls += 1
#                     step = 0.5 * (k1 + k2)
                
#                 else:  # Euler
#                     # Forward Euler integration
#                     if is_dit:
#                         step = model(x, t)[0].reshape(x.shape)
#                         model_calls += 1
#                     else:
#                         step = model(x, t)[0].reshape(x.shape)
#                         model_calls += 1
                
#                 # Update state
#                 x = x + actual_dt * step
#                 t = t + actual_dt
                
#                 if traj:
#                     X.append(x)
    
#     if traj:
#         return x, torch.stack(X), model_calls
#     else:
#         return x, model_calls

import torch
from torchdiffeq import odeint

def integrate_ode(model, x, dt, nt, method='rk4', base_model='resnet', 
                  traj=False, rtol=1e-5, atol=1e-5, time_convention='standard'):
    """
    Integrate ODE using the specified numerical method.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model for computing vector field.
    x : torch.Tensor
        Initial state.
    dt : float
        Time step size (absolute value).
    nt : int
        Number of time steps.
    method : str
        Integration method ('rk4', 'rk2', 'euler', 'dopri5').
    base_model : str
        Model type ('dit' or other).
    traj : bool
        Whether to return the full trajectory.
    rtol : float
        Relative tolerance (for adaptive methods).
    atol : float
        Absolute tolerance (for adaptive methods).
    time_convention : str
        'standard': t=0 is noise, t=1 is data (reversed convention)   
        'reverse': t=1 is noise, t=0 is data (usual Flow Matching convention)
    
    Returns:
    --------
    final_state : torch.Tensor
        Final state after integration.
    trajectory : torch.Tensor (if traj is True)
        The full trajectory of states.
    model_calls : int
        The total number of function evaluations (calls to the model).
    """
    # Set initial and target times based on convention.
    if time_convention == 'reverse':
        # Reverse convention: t=1 is noise, t=0 is data.
        t_start = torch.ones(x.shape[0], device=x.device)
        t_end = 0.0
        actual_dt = -abs(dt)  # Negative step to move toward 0.
    else:
        # Standard convention: t=0 is noise, t=1 is data.
        t_start = torch.zeros(x.shape[0], device=x.device)
        t_end = 1.0
        actual_dt = abs(dt)  # Positive step to move toward 1.
    
    # These variables are used in the non-adaptive (fixed-step) branch.
    t = t_start.clone()
    X = [x] if traj else None
    current_x = x
    model_calls = 0
    
    with torch.no_grad():
        if method == 'dopri5':
            # --- Use torchdiffeq's odeint with the dopri15 solver ---
            # Here we build a time grid from t_start to t_end.
            if time_convention == 'reverse':
                t_grid = torch.linspace(1.0, 0.0, nt+1, device=x.device)
            else:
                t_grid = torch.linspace(0.0, 1.0, nt+1, device=x.device)
            
            # We'll count the number of function evaluations in a mutable container.
            counter = {'calls': 0}
            # def ode_func(t_val, x_val):
            #     counter['calls'] += 1
            #     # In your original code, the model is called as model(x, t)[0] and then reshaped.
            #     return model(x_val, t_val)[0].reshape(x_val.shape)
            
            def ode_func(t_val, x_val):
                counter['calls'] += 1
                # Handle scalar or 0-dim tensor by expanding it to match batch size
                if isinstance(t_val, float) or t_val.dim() == 0:
                    t_tensor = torch.full((x_val.shape[0],), t_val, device=x_val.device)
                else:
                    t_tensor = t_val
                return model(x_val, t_tensor)[0].reshape(x_val.shape)

            
            # Call the adaptive solver (note: using method 'dopri15' from torchdiffeq).
            sol = odeint(ode_func, x, t_grid, method='dopri5', rtol=rtol, atol=atol)
            current_x = sol[-1]
            model_calls = counter['calls']
            if traj:
                # The solution 'sol' is a tensor of shape (nt+1, ...).
                X = sol
        else:
            # --- Use original fixed-step methods (RK4, RK2, Euler) ---
            for i in range(nt):
                # Determine if the model uses the 'dit' convention.
                is_dit = base_model == 'dit'
                if method == 'rk4':
                    if is_dit:
                        k1 = model(x, t)[0].reshape(x.shape)
                        model_calls += 1
                        k2 = model(x + actual_dt/2 * k1, t + actual_dt/2)[0].reshape(x.shape)
                        model_calls += 1
                        k3 = model(x + actual_dt/2 * k2, t + actual_dt/2)[0].reshape(x.shape)
                        model_calls += 1
                        k4 = model(x + actual_dt * k3, t + actual_dt)[0].reshape(x.shape)
                        model_calls += 1
                    else:
                        k1 = model(x, t)[0].reshape(x.shape)
                        model_calls += 1
                        k2 = model(x + actual_dt/2 * k1, t + actual_dt/2)[0].reshape(x.shape)
                        model_calls += 1
                        k3 = model(x + actual_dt/2 * k2, t + actual_dt/2)[0].reshape(x.shape)
                        model_calls += 1
                        k4 = model(x + actual_dt * k3, t + actual_dt)[0].reshape(x.shape)
                        model_calls += 1
                    step = 1/6 * (k1 + 2*k2 + 2*k3 + k4)
                
                elif method == 'rk2':
                    if is_dit:
                        k1 = model(x, t)[0].reshape(x.shape)
                        model_calls += 1
                        k2 = model(x + actual_dt * k1, t + actual_dt)[0].reshape(x.shape)
                        model_calls += 1
                    else:
                        k1 = model(x, t)[0].reshape(x.shape)
                        model_calls += 1
                        k2 = model(x + actual_dt * k1, t + actual_dt)[0].reshape(x.shape)
                        model_calls += 1
                    step = 0.5 * (k1 + k2)
                
                else:  # Euler method.
                    if is_dit:
                        step = model(x, t)[0].reshape(x.shape)
                        model_calls += 1
                    else:
                        step = model(x, t)[0].reshape(x.shape)
                        model_calls += 1
                
                # Update the state and time.
                x = x + actual_dt * step
                t = t + actual_dt
                
                if traj:
                    X.append(x)
            current_x = x
    
    # Return outputs in the expected format.
    if traj:
        # In the torchdiffeq branch, X is already a tensor;
        # in the fixed-step branch, we stack the list.
        if isinstance(X, list):
            return current_x, torch.stack(X), model_calls
        else:
            return current_x, X, model_calls
    else:
        return current_x, model_calls
    

def generate_flow_matched_images(vel_net, decoder, num_samples, batch_size, save_dir, device,
                                latent_channels=4, latent_size=32, nsteps=10,
                                base_model='resnet', int_method='rk4', generator=None, time_convention='standard', return_NFE=False, seed=None):
    """
    Generate images using a trained flow matching model with improved memory management.
    """
    #### NEW ####
    # Seed everything for reproducibility
    if seed is not None:
        # this might have already been set in the main script
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        ############

    import gc
    os.makedirs(save_dir, exist_ok=True)
    generated = 0
    nfe_list = []
    try:
        with torch.no_grad():
            while generated < num_samples:
                # Clear cache at the start of each batch
                torch.cuda.empty_cache()
                
                current_bs = min(batch_size, num_samples - generated)
                print(f"Processing batch {generated//batch_size + 1}, generating {current_bs} images...")
                
                # Generate latent vectors
                z0 = torch.randn(current_bs, latent_channels, latent_size, latent_size, 
                                device=device, generator=generator)
                
                # Integrate using selected method
                dt = 1.0 / nsteps
                if return_NFE==False:
                    z_final, _ = integrate_ode(
                        vel_net, z0, dt, nsteps,
                        method=int_method,
                        base_model=base_model,
                        traj=False, time_convention=time_convention)
                else:
                    z_final, nfe = integrate_ode(
                        vel_net, z0, dt, nsteps,
                        method=int_method,
                        base_model=base_model,
                        traj=False, time_convention=time_convention)
                    nfe_list.append(nfe)

                # Release z0 immediately
                del z0
                torch.cuda.empty_cache() #uncomment if running into memory errors 
                
                # Process in smaller chunks for less memory pressure
                chunk_size = min(64, current_bs)  # Smaller chunks for decoding
                for chunk_idx in range(0, current_bs, chunk_size):
                    end_idx = min(chunk_idx + chunk_size, current_bs)
                    chunk_size_actual = end_idx - chunk_idx
                    
                    # Get chunk of latents
                    z_chunk = z_final[chunk_idx:end_idx]
                    
                    # Decode the latents to images
                    x_chunk = decoder(z_chunk)
                    
                    # Release z_chunk immediately 
                    del z_chunk
                    
                    # Normalize and move to CPU immediately
                    x_chunk = (x_chunk + 1) / 2  # Scale from [-1, 1] to [0, 1]
                    x_chunk = x_chunk.cpu()  # Move to CPU
                    
                    # Save each image in the chunk
                    for i in range(chunk_size_actual):
                        img_idx = generated + chunk_idx + i
                        save_image(x_chunk[i], os.path.join(save_dir, f"gen_{img_idx}.png"))
                    
                    # Release chunk data
                    del x_chunk #uncomment if running into memory errors
                    torch.cuda.empty_cache()  #uncomment if running into memory errors
                
                # Release z_final
                del z_final
                
                # Update generated count
                generated += current_bs
                
                # Force cleanup
                torch.cuda.empty_cache() #uncomment if running into memory errors
                gc.collect() #uncomment if running into memory errors
                
                print(f"Generated {generated}/{num_samples} images")

            torch.cuda.empty_cache()  
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"CUDA out of memory error encountered after generating {generated} images.")
            print("You might need to reduce batch_size or use a smaller model.")
        raise e

    if return_NFE == True:
        return save_dir, nfe_list
    else:
        return save_dir, None

# def generate_flow_matched_images(vel_net, decoder, num_samples, batch_size, save_dir, device, 
#                                 latent_channels=16, latent_size=8, nsteps=200, 
#                                 base_model='resnet', int_method='rk4'):
#     """Generate flow-matched images using the velocity model"""
#     os.makedirs(save_dir, exist_ok=True)
    
#     generated = 0
#     with torch.no_grad():
#         while generated < num_samples:
#             current_bs = min(batch_size, num_samples - generated)
            
#             # Generate latent vectors
#             z0 = torch.randn(current_bs, latent_channels, latent_size, latent_size, device=device)
            
#             # Integrate using selected method
#             dt = 1.0 / nsteps
#             z_final = integrate_ode(
#                 vel_net, z0, dt, nsteps, 
#                 method=int_method, 
#                 base_model=base_model, 
#                 traj=False
#             )
            
#             # Decode the latents to images
#             x = decoder(z_final)
            
#             # Normalize to [0, 1] for saving
#             x = (x + 1) / 2  # Assuming the output is in [-1, 1] range
#             # x = torch.clamp(x, 0.0, 1.0)  # Explicitly clamp
            
#             # Save images
#             for i in range(current_bs):
#                 save_image(x[i], os.path.join(save_dir, f"gen_{generated+i}.png"))
            
#             generated += current_bs
#             print(f"Generated {generated}/{num_samples} flow-matched images")
    
#     return save_dir




# def load_vae(args, load_encoder=True,load_decoder=True):
#     """
#     Load either custom VAE or Hugging Face VAE based on command-line arguments.
    
#     Parameters:
#     -----------
#     args : argparse.Namespace
#         Command-line arguments
        
#     Returns:
#     --------
#     encoder, decoder : tuple of nn.Module
#         Encoder and decoder models
#     """
#     if args.use_hf_vae:
#         print(f"Using Hugging Face VAE: {args.hf_vae_model}")
#         # Check if we already have saved wrappers
#         enc_path = os.path.join(args.model_savepath, "hf_encoder_wrapper.pt")
#         dec_path = os.path.join(args.model_savepath, "hf_decoder_wrapper.pt")
#         enc = None
#         dec = None
#         if os.path.exists(enc_path) and os.path.exists(dec_path):
#             print("Loading saved HF VAE wrappers")
#             if load_encoder==True:
#                 print("Loading saved HF VAE wrapper for encoder")
#                 enc = torch.load(enc_path, map_location=args.device, weights_only=False)
        
#             if load_decoder==True:
#                 print("Loading saved HF VAE wrapper for decoder")
#                 dec = torch.load(dec_path, map_location=args.device, weights_only=False)
#         else:
#             print("Creating new HF VAE wrappers")
#             enc, dec = create_hf_vae_wrappers(pretrained_model_name_or_path=args.hf_vae_model, device=args.device)
            
#             # Save the wrappers directly to the specified save path
#             torch.save(enc, os.path.join(args.model_savepath, "hf_encoder_wrapper.pt"))
#             torch.save(dec, os.path.join(args.model_savepath, "hf_decoder_wrapper.pt"))
#             print(f"HF VAE wrappers saved to: {args.model_savepath}")
#     else:
#         print("Using custom VAE models")
#         try:
#             if load_encoder==True:
#                 print("Loading custom VAE encoder")
#                 enc = torch.load(os.path.join(args.model_savepath, 'encoder.pt'), map_location=args.device, weights_only=False)
#             if load_decoder==True:
#                 print("Loading custom VAE decoder")
#                 dec = torch.load(os.path.join(args.model_savepath, 'decoder.pt'), map_location=args.device, weights_only=False)
#         except FileNotFoundError:
#             raise ValueError("Custom VAE models not found. Please train custom VAE first or use --use_hf_vae")
    
#     return enc, dec

import math, os
import torch
from torchvision.utils import save_image

import math, os, torch
from torchvision.utils import save_image

@torch.no_grad()
def generate_flow_matched_images_conditional(
    vel_net,
    decoder,
    num_samples: int,
    batch_size: int,
    save_dir: str,
    device: str,
    latent_channels: int,
    latent_size: int,
    nsteps: int,
    base_model: str,
    int_method: str = "rk4",          # supports "rk4","rk2","euler","dopri5" in your integrate_ode
    time_convention: str = "reverse", # "reverse" (t:1->0) or "standard" (t:0->1)
    cfg_scale: float = 1.0,           # >1 enables batched CFG path in vel_net during eval
    use_uniform_label_sampling: bool = False,  # if True, sample labels ~ Uniform(0..num_classes-1)
    num_classes: int = None,          # required if use_uniform_label_sampling=True
    labels: torch.Tensor = None,      # optional pre-specified labels (len=num_samples, long)
    return_NFE: bool = False,
    seed: int = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
):
    """
    Conditional sampler compatible with your integrate_ode(model, x, dt, nt, ...).

    - Builds a model wrapper with signature (x, t) -> (v, *extras) expected by integrate_ode,
      injecting labels y and cfg_scale into vel_net.
    - If cfg_scale > 1, keeps batch size even (batched CFG assumption).
    - Supports dopri5 via your integrate_ode.
    - Uniform class sampling across ImageNet classes for FID if requested.
    """
    os.makedirs(save_dir, exist_ok=True)

    # RNG for reproducibility (also used in torch.randn(..., generator=g))
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
    else:
        g = None

    # ----- assemble label vector for all samples -----
    if labels is not None:
        assert len(labels) == num_samples, "labels length must match num_samples"
        all_labels = labels.clone().long().to(device)
    elif use_uniform_label_sampling:
        assert num_classes is not None and num_classes > 0, \
            "num_classes must be provided for uniform label sampling."
        all_labels = torch.randint(0, num_classes, (num_samples,),
                                   dtype=torch.long, device=device, generator=g)
    else:
        # default to class 0 if nothing is provided
        all_labels = torch.zeros(num_samples, dtype=torch.long, device=device)

    # --- ensure eval so CFG paths in vel_net are active ---
    was_training = vel_net.training
    vel_net.eval()

    saved_paths = []
    total_nfe = 0
    img_counter = 0

    # The integrate_ode controls the time bounds internally via time_convention.
    # We only need to provide dt (= 1/nsteps) and nt (= nsteps).
    dt = 1.0 / max(1, nsteps)

    # Helper module to adapt vel_net to (x,t) -> (velocity, *extras) signature
    class _ModelWithCond(torch.nn.Module):
        def __init__(self, vel_net, y_batch, cfg_scale):
            super().__init__()
            self.vel_net = vel_net
            self.y = y_batch
            self.cfg_scale = float(cfg_scale)
        def forward(self, x, t):
            # t can be scalar (0-d) or [B]; make sure it’s [B]
            if isinstance(t, float) or (torch.is_tensor(t) and t.dim() == 0):
                t = torch.full((x.shape[0],), float(t), device=x.device)
            v, _, _ = self.vel_net(x, t, y=self.y, cfg_scale=self.cfg_scale)
            # integrate_ode expects to index [0] from the return
            return (v,)

    # Choose base_model flag for integrate_ode (kept for compatibility; behavior is same)
    base_flag = 'dit' if 'DiT' in vel_net.reaction.__class__.__name__ else base_model

    n_batches = math.ceil(num_samples / batch_size)
    for b in range(n_batches):
        b_start = b * batch_size
        b_end   = min(num_samples, (b + 1) * batch_size)
        B = b_end - b_start
        if B <= 0:
            break

        # keep batch even if using batched CFG
        use_cfg = (cfg_scale is not None and cfg_scale > 1.0)
        if use_cfg and (B % 2 == 1):
            # simplest: drop one sample this batch (still reaches total num_samples over all batches)
            b_end -= 1
            B -= 1
            if B <= 0:
                continue

        y_batch = all_labels[b_start:b_end]              # [B]
        z0 = torch.randn(B, latent_channels, latent_size, latent_size,
                         device=device, generator=g)      # noise in latent space

        model_wrapper = _ModelWithCond(vel_net, y_batch, cfg_scale)

        # call integrate_ode; it returns (final_x, model_calls) when traj == False
        xT, nfe = integrate_ode(
            model=model_wrapper,
            x=z0,
            dt=dt,
            nt=nsteps,
            method=int_method,
            base_model=base_flag,
            traj=False,
            rtol=rtol,
            atol=atol,
            time_convention=time_convention
        )
        if isinstance(nfe, (int, float)):
            total_nfe += nfe

        # decode and clamp to [-1,1]
        imgs = decoder(xT)
        imgs = torch.clamp(imgs, -1.0, 1.0)

        # save each image
        for i in range(B):
            fn = os.path.join(save_dir, f"{img_counter:06d}.png")
            save_image(imgs[i] * 0.5 + 0.5, fn)  # map [-1,1] to [0,1]
            saved_paths.append(fn)
            img_counter += 1

    # restore train/eval
    if was_training:
        vel_net.train()

    used_labels = all_labels[:img_counter].detach().cpu()
    if return_NFE:
        return save_dir, used_labels, total_nfe
    return save_dir, used_labels



def return_RK4_functions(base_model):
    if base_model=='dit':
        # DiT 
        def RK4step(model, x, t, dt):        
            k1 = model(t, x)[0].reshape(x.shape)
            k2 = model(t+dt/2, x+dt/2*k1)[0].reshape(x.shape)
            k3 = model(t+dt/2, x+dt/2*k2)[0].reshape(x.shape)
            k4 = model(t+dt, x+dt*k3)[0].reshape(x.shape)
            step = 1/6*(k1 + 2*k2 + 2*k3 + k4)
            return step

        def RK4(model, x, dt, nt):
            # this RK4 function makes the model take time first, as expected by DiT 
            t = torch.zeros(x.shape[0], device=x.device)
            X = [x]
            with torch.no_grad():
                for i in range(nt):
                    step = RK4step(model, x, t, dt)
                    x  = x + dt*step
                    t = t + dt
                    X.append(x)
            return x,  torch.stack(X)
        return RK4, RK4step
    else:
        # Define the RK4 step and integration functions from original code
        def RK4step(model, x, t, dt):        
            k1 = model(x, t)[0].reshape(x.shape)
            k2 = model(x+dt/2*k1, t+dt/2)[0].reshape(x.shape)
            k3 = model(x+dt/2*k2, t+dt/2)[0].reshape(x.shape)
            k4 = model(x+dt*k3, t+dt)[0].reshape(x.shape)
            step = 1/6*(k1 + 2*k2 + 2*k3 + k4)
            return step

        def RK4(model, x, dt, nt):
            t = torch.zeros(x.shape[0], device=x.device)
            X = [x]
            with torch.no_grad():
                for i in range(nt):
                    step = RK4step(model, x, t, dt)
                    x = x + dt*step
                    t = t + dt
                    X.append(x)
            return x, torch.stack(X)
    
        return RK4, RK4step
    

def generate_experiment_name(args):
    """
    Generate a concise experiment name from key hyperparameters.
    This name includes the dataset, nt_max, and nsteps values.
    """
    return f"{args.dataset}_nt{args.nt_max}_nsteps{args.nsteps}"


# Helper function to get the model directory path
def get_model_dir(base_path, flow_model_type, base_model=None):
    """
    Returns the model directory path based on the model type and base model.
    
    Parameters:
    -----------
    base_path : str
        Base directory path where models are stored
    flow_model_type : str
        Type of flow model (e.g., 'diffusion_reaction', 'hybrid_gnn')
    base_model : str, optional
        Base model type (e.g., 'resnet', 'dit')
        
    Returns:
    --------
    str : Path to the model directory
    """
    # Simply return the base path without creating subfolders
    return base_path


# Helper function to save model
def save_model(model, base_path, flow_model_type, base_model=None, filename="vel_net.pt"):
    """
    Saves a model to the appropriate directory.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to save
    base_path : str
        Base directory path where models are stored
    flow_model_type : str
        Type of flow model (e.g., 'diffusion_reaction', 'hybrid_gnn')
    base_model : str, optional
        Base model type (e.g., 'resnet', 'dit')
    filename : str, optional
        Name of the file to save the model to (default: "vel_net.pt")
    """
    # Get the model directory (which is just the base path)
    model_dir = get_model_dir(base_path, flow_model_type, base_model)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(model_dir, filename)
    torch.save(model, model_path)
    print(f"Model saved to: {model_path}")
    
    return model_path

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def match_points(x0, x1):
    D = torch.cdist(x0, x1)
    I = torch.argmin(D, dim=1)
    J = torch.argmin(D, dim=0)
    x1 = torch.cat((x1, x1[I,:]), dim=0)
    x0 = torch.cat((x0[J,:], x0), dim=0)
    return x0, x1

def match_points_cat(x0, x1):
    x02 = x0.view(x0.shape[0],-1)
    x12 = x1.view(x1.shape[0],-1)
    D = torch.cdist(x02, x12)
    I = torch.argmin(D, dim=1)
    J = torch.argmin(D, dim=0)
    x12 = torch.cat((x12, x12[I,:]), dim=0)
    x02 = torch.cat((x02[J,:], x02), dim=0)

    x02 = x02.reshape(x0.shape[0]+x0.shape[0], x0.shape[1], x0.shape[2], x0.shape[-1])
    x12 = x12.reshape(x1.shape[0]+x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[-1])
    return x02, x12

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


def RK4step_DiT(model, x, t, dt):        
    k1 = model(t, x).reshape(x.shape)
    k2 = model(t+dt/2, x+dt/2*k1).reshape(x.shape)
    k3 = model(t+dt/2, x+dt/2*k2).reshape(x.shape)
    k4 = model(t+dt, x+dt*k3).reshape(x.shape)
    step = 1/6*(k1 + 2*k2 + 2*k3 + k4)
    return step

def RK4_DiT(model, x, dt, nt):
    # this RK4 function makes the model take time first, as expected by DiT 
    t = torch.zeros(x.shape[0], device=x.device)
    X = [x]
    with torch.no_grad():
        for i in range(nt):
            step = RK4step_DiT(model, x, t, dt)
            x  = x + dt*step
            t = t + dt
            X.append(x)
    return x,  torch.stack(X)


###### IMAGE functions #######


# Add these imports at the top of your script with the other imports
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import math 
# Add this function definition before your main code blocks
def plot_images_grid(image_folder, output_pdf_path):
    """
    Create a PDF with a grid layout of all images from a folder.
    
    Parameters:
    -----------
    image_folder : str
        Path to the folder containing images to plot
    output_pdf_path : str
        Path where the output PDF will be saved
    
    Returns:
    --------
    None
    """
    # Get all image files from the folder
    image_files = [f for f in os.listdir(image_folder)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    # Sort the files if needed
    image_files.sort()
    
    # Calculate the optimal grid dimensions for a square-like layout
    n_images = len(image_files)
    n_cols = int(math.ceil(math.sqrt(n_images)))
    n_rows = int(math.ceil(n_images / n_cols))
    
    print(f"Arranging {n_images} images in a {n_rows}x{n_cols} grid")
    
    # Create a PDF file
    with PdfPages(output_pdf_path) as pdf:
        # Create figure with tight layout to minimize whitespace
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        
        # Make axes a 2D array even if we have only one row or column
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot each image
        for i, image_file in enumerate(image_files):
            if i < n_images:
                # Calculate row and column indices
                row_idx = i // n_cols
                col_idx = i % n_cols
                
                # Load and plot the image
                img_path = os.path.join(image_folder, image_file)
                img = np.array(Image.open(img_path))
                axes[row_idx, col_idx].imshow(img)
                axes[row_idx, col_idx].set_axis_off()
        
        # Hide axes for empty spots in the grid
        for i in range(n_images, n_rows * n_cols):
            row_idx = i // n_cols
            col_idx = i % n_cols
            axes[row_idx, col_idx].set_visible(False)
        
        # Ensure tight layout and save to PDF
        plt.tight_layout(pad=0.1)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"PDF saved to {output_pdf_path}")


def plot_tensor_panel(batch):
    b = batch.shape[0]
    rows = 1 # int(np.sqrt(b))
    cols = b//rows 
    batch = batch[:rows*cols]
    
    # Convert tensor to numpy for plotting
    batch_np = batch.permute(0, 2, 3, 1).cpu().numpy()  # Shape: (B, H, W, C)
    
    # Normalize images to [0, 1] range for visualization
    batch_np = (batch_np - batch_np.min()) / (batch_np.max() - batch_np.min())
    
    # Create the panel
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    else:
        axes = np.array(axes).flatten()
    
    for i in range(rows * cols):
        ax = axes[i]
        ax.imshow(batch_np[i])
        ax.axis('off')
    
    plt.tight_layout(pad=0.5)
    return fig  # return the figure object instead of calling plt.show()


def plot_reconstruction_comparison(original_imgs, reconstructed_imgs, max_images=8, rescale=False):
    """
    Plot a high-quality comparison of original and reconstructed images.
    
    Args:
        original_imgs: Tensor of original images [B, C, H, W]
        reconstructed_imgs: Tensor of reconstructed images [B, C, H, W]
        max_images: Maximum number of image pairs to display
        rescale: If True, rescale images from [-1,1] to [0,1] for visualization
    
    Returns:
        matplotlib figure object
    """
    # Limit the number of images to display
    num_images = min(original_imgs.shape[0], max_images)
    original_imgs = original_imgs[:num_images]
    reconstructed_imgs = reconstructed_imgs[:num_images]
    
    # Create figure with 2 rows (original & reconstruction) and n columns (images)
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2.5, 5))
    
    # Convert tensors to numpy
    orig_np = original_imgs.permute(0, 2, 3, 1).cpu().detach().numpy()
    recon_np = reconstructed_imgs.permute(0, 2, 3, 1).cpu().detach().numpy()
    
    # Rescale from [-1,1] to [0,1] if needed
    if rescale:
        orig_np = (orig_np + 1) / 2
        recon_np = (recon_np + 1) / 2
    
    # Ensure values are in valid range for imshow
    orig_np = np.clip(orig_np, 0, 1)
    recon_np = np.clip(recon_np, 0, 1)
    
    # Handle single image case
    if num_images == 1:
        axes = axes.reshape(2, 1)
    
    # Plot original images in the top row
    for i in range(num_images):
        axes[0, i].imshow(orig_np[i])
        axes[0, i].set_title('Original', fontsize=12)
        axes[0, i].axis('off')
    
    # Plot reconstructed images in the bottom row
    for i in range(num_images):
        axes[1, i].imshow(recon_np[i])
        axes[1, i].set_title('Reconstructed', fontsize=12)
        axes[1, i].axis('off')
    
    # Adjust layout to minimize whitespace
    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    fig.tight_layout(pad=0.5)
    
    return fig


def plot_side_by_side_comparison(original_imgs, reconstructed_imgs, max_pairs=6, rescale=False):

    """
    Plot original and reconstructed images side by side for easier one-to-one comparison.
    
    Args:
        original_imgs: Tensor of original images [B, C, H, W]
        reconstructed_imgs: Tensor of reconstructed images [B, C, H, W]
        max_pairs: Maximum number of image pairs to display
        rescale: If True, rescale images from [-1,1] to [0,1] for visualization
    
    Returns:
        matplotlib figure object
    """
    # Limit the number of images to display
    num_pairs = min(original_imgs.shape[0], max_pairs)
    original_imgs = original_imgs[:num_pairs]
    reconstructed_imgs = reconstructed_imgs[:num_pairs]
    
    # Configure figure with two columns per pair
    fig, axes = plt.subplots(num_pairs, 2, figsize=(6, num_pairs * 3))
    
    # Convert tensors to numpy
    orig_np = original_imgs.permute(0, 2, 3, 1).cpu().detach().numpy()
    recon_np = reconstructed_imgs.permute(0, 2, 3, 1).cpu().detach().numpy()
    
    # Rescale from [-1,1] to [0,1] if needed
    if rescale:
        orig_np = (orig_np + 1) / 2
        recon_np = (recon_np + 1) / 2
        
    # Ensure values are in valid range for imshow
    orig_np = np.clip(orig_np, 0, 1)
    recon_np = np.clip(recon_np, 0, 1)
    
    # Handle single pair case
    if num_pairs == 1:
        axes = axes.reshape(1, 2)
    
    # Plot each pair of images
    for i in range(num_pairs):
        axes[i, 0].imshow(orig_np[i])
        axes[i, 0].set_title('Original', fontsize=12)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(recon_np[i])
        axes[i, 1].set_title('Reconstructed', fontsize=12)
        axes[i, 1].axis('off')
    
    # Adjust spacing to minimize whitespace
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    fig.tight_layout(pad=0.5)
    
    return fig



############### TRAINING UTILS 


def monitor_adj_generator_training(vel_net, iteration, log_every=10):
    """
    Monitor if the adjacency generator is training by tracking parameter and gradient norms.
    """
    stats = {}
    
    # Navigate to the adjacency generator
    nonlinear_heat = vel_net.diffusion_term
    adj_generator = nonlinear_heat.adj_generator
    
    # Calculate parameter norm
    param_norm = 0
    for name, param in adj_generator.named_parameters():
        if param.requires_grad:
            param_norm += param.norm().item() ** 2
    param_norm = param_norm ** 0.5
    stats['param_norm'] = param_norm
    
    # Calculate gradient norm
    grad_norm = 0
    grad_exists = False
    for name, param in adj_generator.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm += param.grad.norm().item() ** 2
            grad_exists = True
    
    grad_norm = grad_norm ** 0.5 if grad_exists else 0
    stats['grad_norm'] = grad_norm
    
    # Print detailed information periodically (if log_every > 0)
    if log_every > 0 and iteration % log_every == 0:
        print(f"\nIteration {iteration} - Adjacency Generator Stats:")
        print(f"  Parameter norm: {param_norm:.6f}")
        print(f"  Gradient norm: {grad_norm:.6f}")
        
        # More detailed parameter inspection
        if log_every > 0 and iteration % (log_every * 10) == 0:
            print("\nDetailed Parameter Info:")
            for name, param in adj_generator.named_parameters():
                if param.requires_grad:
                    grad_info = f"grad_norm: {param.grad.norm().item():.6f}" if param.grad is not None else "no grad"
                    print(f"  {name}: param_norm: {param.norm().item():.6f}, {grad_info}")
    
    return stats



# Define parameter groups with different learning rates
# Define parameter groups with different learning rates
def create_param_groups(model, adj_generator_lr=1e-2, base_lr=3e-4):
    # First find the adjacency generator
    nonlinear_heat = model.diffusion_term
    adj_generator = nonlinear_heat.adj_generator
    
    # Create two parameter groups
    adj_params = []
    other_params = []
    
    # Loop through all parameters
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Check if this parameter belongs to the adjacency generator
        is_adj_param = False
        for adj_name, adj_param in adj_generator.named_parameters():
            if param is adj_param:  # Compare by identity
                is_adj_param = True
                break
                
        if is_adj_param:
            adj_params.append(param)
        else:
            other_params.append(param)
    
    # Create parameter groups
    param_groups = [
        {'params': adj_params, 'lr': adj_generator_lr},
        {'params': other_params, 'lr': base_lr}
    ]
    
    print(f"Adjacency generator parameters: {len(adj_params)}")
    print(f"Other parameters: {len(other_params)}")
    
    return param_groups



########### Constrained ################

import torch

def zero_right_half(images) :
    """
    Zero out the right half of each image in a batch.
    
    Args:
        images (torch.Tensor): Batch of images of shape [B, C, H, W].
    
    Returns:
        torch.Tensor: The masked batch of images, same shape as input.
    """
    # images: [B, C, H, W]
    B, C, H, W = images.shape
    # compute midpoint along width
    mid = W // 2
    # create a horizontal mask of shape [1, 1, 1, W] with ones on left half, zeros on right
    mask = (torch.arange(W, device=images.device) < mid) \
               .to(images.dtype) \
               .reshape(1, 1, 1, W)
    # broadcast mask over (B, C, H) and multiply
    return images * mask


def zero_right_half(images) :
    """
    Zero out the right half of each image in a batch.
    
    Args:
        images (torch.Tensor): Batch of images of shape [B, C, H, W].
    
    Returns:
        torch.Tensor: The masked batch of images, same shape as input.
    """
    # images: [B, C, H, W]
    B, C, H, W = images.shape
    # compute midpoint along width
    mid = W // 2
    # create a horizontal mask of shape [1, 1, 1, W] with ones on left half, zeros on right
    mask = (torch.arange(W, device=images.device) < mid) \
               .to(images.dtype) \
               .reshape(1, 1, 1, W)
    # broadcast mask over (B, C, H) and multiply
    return images * mask


import torch

def extract_except_top_left_patch(images: torch.Tensor, percent: float) -> torch.Tensor:
    """
    Extract all pixels except the top-left square patch.

    Args:
        images (torch.Tensor): Input tensor of shape (B, C, H, W).
        percent (float): Either
            - a fraction in [0, 1], or
            - a percentage in [0, 100];
          denoting the *side-length* of the square patch to drop,
          as a fraction of the *shorter* image dimension.

    Returns:
        torch.Tensor: Shape (B, C, H*W - N*N), containing the remaining pixels
        in row-major order (first the top-right region, then the entire bottom).
    """
    B, C, H, W = images.shape

    # interpret percent as [0,1] fraction
    frac = percent / 100.0 if percent > 1.0 else percent
    frac = float(max(0.0, min(frac, 1.0)))

    # side length of square patch to exclude
    N = int(min(H, W) * frac)

    # slice out the two complementary regions:
    # 1) the top-right:    rows 0:N, cols N:W
    # 2) the entire bottom: rows N:H, cols 0:W
    top_right = images[:, :, :N, N:].reshape(B, C, -1)
    bottom    = images[:, :, N:,  : ].reshape(B, C, -1)

    # concatenate along the “pixel” dimension
    return torch.cat((top_right, bottom), dim=2)



# from torch.autograd.functional import jvp, vjp  # this requires backwards of backwards trick, so use torch.func instead
from torch.func import jvp, vjp
import torch.nn as nn

class ConstrainModel(nn.Module):
    # REVERSE time convention
    def __init__(self, original_model, decoder):
        super().__init__()
        self.model = original_model
        self.decoder = decoder

    def constraint_function(self, xt):
        with torch.enable_grad():
            yt = self.decoder(xt)
            yt = zero_right_half(yt)

        return yt
    
    # def JTJ_lambda(self, x, lambda_):
    #     _, jt = torch.autograd.functional.vjp(self.constraint_function, x, lambda_)
    #     _, jtj = torch.autograd.functional.jvp(self.constraint_function, x, jt) #Jjt * lambda
    #     return jtj

    def JJT_lambda(self, x):
        # 1) one shot: run the forward pass and grab the pullback function  
        #    pullback(v) = J^T v
        with torch.enable_grad():
            # pullback is a callable that maps cotangent->cotangent
            _, pullback = vjp( self.constraint_function, x)

        # 2) return a matvec that applies JJ^T·lambda for any lambda
        def matvec(lam):
            (jt_lam, ) = pullback(lam) # J^T lambda
            _, jjt_lam = jvp( self.constraint_function, (x, ), (jt_lam, ) ) # J_J^T(x) @ lambda

            return jjt_lam.detach() 

        return matvec

    def conjugate_gradient(self, matvec, b, x0=None, tol=1e-6, maxiter=50):
        """
        Solve A x = b for x using CG, keeping everything in
        the same shape as b (e.g. [B,C,H,W]).
        """

        # 1) initialize
        x = torch.zeros_like(b) if x0 is None else x0.clone()
        # (temp, ) = matvec(x)
        r = b - matvec(x).detach()                         # same shape as b
        p = r.clone()                                # same shape
        rsold = torch.sum(r * r)                     # scalar
        for i in range(maxiter):
            Ap = matvec(p)  
            with torch.no_grad():                        # same shape as b
                Ap = Ap.detach()  
                if torch.sum(p*Ap) == 0:
                    break
                alpha = rsold / torch.sum(p * Ap)        # scalar
                x = x + alpha * p                        # shape preserved
                r = r - alpha * Ap                       # shape preserved
                rsnew = torch.sum(r * r)                 # scalar
                if torch.sqrt(rsnew) < tol:
                    break
                p = r + (rsnew / rsold) * p              # shape preserved
                rsold = rsnew

        return x.detach()

    def forward(self, x, t, initial_noise, data, CG_tol=1e-6, CG_maxiter=50):

        # _, jvp = torch.autograd.functional.jvp(self.constraint_function, x, w) # J_c(x) \times w
        # _, vjp = torch.autograd.functional.vjp(self.constraint_function, x, v) # J_c(x)^T \times v

        # data        = data.detach().requires_grad_(True)
        # initial_noise = initial_noise.detach().requires_grad_(True)
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            
        with torch.no_grad():
            y1 = data
            y0 = self.constraint_function(initial_noise)
            v_theta = self.model(x,t)[0]

        with torch.enable_grad():
            ## Define JTJ for current x
            # JTJ_lam = lambda lam: self.JTJ_lambda(x, lam)
            ## b = y0 - y1 - J_c(x) @ v_theta
            # _, J_v_theta = torch.autograd.functional.jvp(self.constraint_function, x, v_theta) # this needs the backwards of backwards trick, so use torch.func instead

            JJT_function = self.JJT_lambda(x)    # caches constraint_function(x) & pullback
        
            _, J_v_theta = jvp(self.constraint_function, (x,), (v_theta,))

            _, pullback = vjp( self.constraint_function, x)

        J_v_theta = J_v_theta.detach()

        b = (1-1e-5)*y0 - y1 - J_v_theta.detach()

        with torch.no_grad():

            ##### TEST ###
            # for lam in [torch.randn_like(b) for _ in range(10)]:
            #     v = JJT_function(lam)
            #     # should be non-negative:
            #     q = (lam*v).sum().item()   
            #     print("lamᵀ A lam =", q)

            # for lam in torch.randn(10,3,256,256, device=x.device):  
            #     lam = lam.unsqueeze(0)
            #     q = (lam * JJT_function(lam)).sum().item()
            #     print("lamᵀ A lam =", q)

            #############
            lambda_sol = self.conjugate_gradient(JJT_function, b, tol=CG_tol, maxiter=CG_maxiter)

            # _, JT_lam = torch.autograd.functional.vjp(self.constraint_function, x, lambda_sol)

            (JT_lam, ) = pullback(lambda_sol)
            
        output_v =  v_theta.detach() + JT_lam.detach()

        return output_v
    
