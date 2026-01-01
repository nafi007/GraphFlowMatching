import os
import sys
# Add the parent directory of folder1 to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from matplotlib import pyplot as plt
import math
# Import your existing components
from datasets.datasets_256 import AFHQInMemoryDataset_256, PreEncoded_AFHQ_256_Dataset, CelebAHQ256Dataset, PreEncoded_CelebAHQ256_Dataset, PreEncoded_LSUNChurch_Dataset, LSUNChurchDataset

from networks.eldad_networks import resnet, Encoder, Decoder, ASPP
from utils.utils import plot_tensor_panel, plot_images_grid, return_RK4_functions, load_vae, generate_flow_matched_images, set_seed, ensure_vae_wrappers, setup_training_components
from networks.pnp_flow_UNet import UNet2
# Import the new hybrid models
from networks.hybrid_networks import create_hybrid_model

# NEW IMPORTS: For Hugging Face VAE integration
from diffusers import AutoencoderKL
from torchvision.utils import save_image, make_grid
from networks.networks import EncoderWrapper, DecoderWrapper
torch.serialization.add_safe_globals([DecoderWrapper])
torch.serialization.add_safe_globals([EncoderWrapper])
from utils.EMA import EMA
# FID imports
# from FID_optimization import add_fid_args, optimize_fid_directly
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import argparse
from matplotlib import pyplot as plt
import math
import random
from networks.networks import create_hf_vae_wrappers
# Add this to your imports at the top of the file
from torchvision.utils import save_image, make_grid
from datasets.lsun_datasets import LSUNBedroomDataset
from cleanfid import fid
import numpy as np
from utils.utils import load_pretrained_dit_model, load_pretrained_unet_weights

## FOR ADM : 
from utils.model_utils import load_adm_unet, load_edm_unet
# config_path = '/home/shahriar/FlowMatchingPredCor/networks/adm_unet_config.yaml'
# # Load the model and configuration
# unet_model, unet_config = load_adm_unet(config_path, device='cuda', use_fp16=False, use_checkpoint=False)


# config_path = '/home/shahriar/FlowMatchingPredCor/networks/adm_unet_config.yaml'
# # Load the model and configuration
# unet_model, unet_config = load_adm_unet(config_path, device='cuda', use_fp16=False, use_checkpoint=False)

# # You can now use 'unet_model' in your training loop.
# print(f"Loaded ADM UNet with {sum(p.numel() for p in unet_model.parameters() if p.requires_grad)} parameters.")
# from ConsistencyFM import NCSNpp, get_model_config_32x32


##################################
########### PARSING: ###########
##################################
parser = argparse.ArgumentParser()

torch.cuda.empty_cache()

# from ConsistencyFM import NCSNpp, get_model_config_32x32

### DATA ####
parser.add_argument('--dataset', type=str, default='lsun_bedrooms', choices=['ffhq','lsun_bedrooms', 'lsun_church', 'celeba-hq', 'AFHQ-Cat-Full-256']) #'lsun_bedrooms', 'AFHQ-Cat-256'
parser.add_argument('--datapath', type=str, default="/home/shahriar/data/lsun", choices=["/home/shahriar/data/FFHQ/FFHQ_256", "/home/shahriar/data/lsun", "/home/shahriar/data/celeba/celeba_hq_256", "/data/shahriar/datasets/celeba/celeba_hq_256", "/home/shahriar/data/afhq_v2"]) #/home/shahriar/data/afhq_v2 #before: /home/shahriar/data/afhq
# Parse arguments - add new arguments for pre-encoded data
parser.add_argument('--use_pre_encoded', action='store_true', 
                    help='Use pre-encoded dataset instead of original AFHQ dataset')
parser.add_argument('--no-use_pre_encoded', dest='use_pre_encoded', action='store_false', 
                    help='Use original AFHQ dataset')
parser.set_defaults(use_pre_encoded=True)

parser.add_argument('--encoded_dataset_path', type=str, default="/data/shahriar/datasets/lsun/lsun_encoded_mse/lsun_bedrooms", choices=["/data/shahriar/datasets/lsun/lsun_encoded_mse/lsun_bedrooms",
                                                                                                                                         "/home/shahriar/data/FFHQ/FFHQ_256_mse_encoded",
                                                                                                                                    "/home/shahriar/data/lsun/lsun_encoded_mse/lsun_bedrooms", 
                                                                                                                                    "/home/shahriar/data/lsun/lsun_encoded_mse/church_train", 
                                                                                                                                    "/home/shahriar/data/celeba/celeba_hq_256_encoded_noFlips",
                                                                                                                                     "/data/shahriar/datasets/celeba/celeba_hq_256_encoded_noFlips", 
                                                                                                                                    '/data/shahriar/datasets/celeba/celeba_hq_256_encoded', 
                                                                                                                                    '/home/shahriar/data/celeba/celeba_hq_256_encoded',
                                                                                                                                    "/home/shahriar/data/afhq_v2_encoded_mse", 
                                                                                                                                    "/data/shahriar/datasets/afhq_v2_encoded_mse"],
                    help='Path to pre-encoded dataset directory')

parser.add_argument('--cleanfid_dataset_name', type=str, default='lsun_bedrooms_256_all', choices=['ffhq_256_all', 'lsun_bedrooms_256_all', 'lsun_church_256_train', 'celeba_256', "afhq_cat_256_train_test_full"], help='Integration method to use')

parser.add_argument('--lsun_bedrooms_dataset_cache_chunks', type=int, default=10, help='Number of chunks to cache for LSUN Bedrooms dataset')
#################################
# Model architecture choices
# For our paper, we kept "flow_model_type" as nonLinearHeatDiffusion2
parser.add_argument('--flow_model_type', type=str, default='nonLinearHeatDiffusion2', 
                    choices=['nonLinearHeatDiffusion2','nonLinearHeatDiffusion', 'diffusion_reaction_6_improved' , "diffusion_reaction_7_withTimeAndAttention", 'nonlinear_heat_eq', 'diffusion_reaction_6', 'diffusion_reaction_5', 'diffusion_reaction_4', 'diffusion_reaction_3'], 
                    help='Type of flow model to use (for diffusion reaction, this is the reaction term in dX/dt)')
parser.add_argument('--base_model', type=str, default='dit', 
                    choices=['adm', 'resnet', 'dit', 'pnpUNet'], 
                    help='Base model for the hybrid architecture')

parser.add_argument('--adj_mode', type=str, default='attention', 
                    choices=['attention', 'cosine', 'gaussian', 'knn'], 
                    help='Method to generate adjacency matrix')
parser.add_argument('--knn_k', type=int, default=20, help='The number of neighbours to choose if adjacency mode is K nearest neighbors (this uses cosine similarity in latent space)')

parser.add_argument('--diffusion_resnet_channels', type=int, default=64, #128
                    help='Number of resnet channels for the diffusion coefficient network')

parser.add_argument('--resnet_layers', type=int, default=3,
                    help='Number of resnet layers')

parser.add_argument('--Diffusion_term_activation_function', type=str, default='ELU', 
                    help='Activation function used for diffusion coefficient. Choices: ELU, Softmax, SiLU, ReLU')

parser.add_argument('--diffusion', action='store_true', help='Use diffusion/Laplacian term')
parser.add_argument('--no-diffusion', dest='diffusion', action='store_false', help='Do not use diffusion/Laplacian term')
parser.set_defaults(diffusion=False)

parser.add_argument('--diffusion_network', type=str, default='NonLinearHeat', choices=['gps', 'NonLinearHeat', 'UNet2', 'dit', 'None']) # 'resnet', 'dit'

# whether or not the Laplacian was an identity for this run (used for tests)
parser.add_argument('--identity_Laplacian', action='store_true', help='Use identity Laplacian for testing')
parser.add_argument('--no-identity_Laplacian', dest='identity_Laplacian', action='store_false', help='Use computed Laplacian')
parser.set_defaults(identity_Laplacian=False)

# Add the standard arguments
parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
parser.add_argument('--no-wandb', dest='use_wandb', action='store_false', help='Disable wandb logging')
parser.set_defaults(use_wandb=False)

parser.add_argument('--nt_max', type=int, default=1)

# parser.add_argument('--model_savepath', type=str, default='/home/shahriar/FlowMatchingPredCor/Cat256/models/HuggingFaceVAE/DiffusionReaction5/PreEncoded/PreEncoded_SiLU_BatchSize256_pnpUNet_diffNet_pnpUNet')  #/home/shahriar/FlowMatchingPredCor/Cat64/models/eldad_vae/DiffusionReaction/diffusion_reaction_resnet
# parser.add_argument('--image_savepath', type=str, default='/home/shahriar/FlowMatchingPredCor/Cat256/generated_images/vae/HuggingFaceVAE/DiffusionReaction5/PreEncoded/PreEncoded_SiLU_BatchSize256_pnpUNet_diffNet_pnpUNet') # '/home/shahriar/FlowMatchingPredCor/Cat64/generated_images/eldad_vae/DiffusionReaction/weighted_laplacian_diffusion_reaction_resnet'

parser.add_argument('--model_savepath', type=str, default='/home/shahriar/FlowMatchingPredCor/LSUN_Church/models/HuggingFaceVAE/default_temp')  #/home/shahriar/FlowMatchingPredCor/Cat64/models/eldad_vae/DiffusionReaction/diffusion_reaction_resnet
parser.add_argument('--image_savepath', type=str, default='/home/shahriar/FlowMatchingPredCor/LSUN_Church/generated_images/default_temp') # '/home/shahriar/FlowMatchingPredCor/Cat64/generated_images/eldad_vae/DiffusionReaction/weighted_laplacian_diffusion_reaction_resnet'

parser.add_argument('--device', type=str, default='cuda:1')

parser.add_argument('--train_batch_size', type=int, default=50) #256 for 13 mil case #115 for 16.5 mil case
parser.add_argument('--num_workers', type=int, default=0) #16

parser.add_argument('--latent_channels', type=int, default=4)  # CHANGED: Default to 4 for SD VAE

parser.add_argument('--patience', type=int, default=20000) # scheduler patience, the 5000 assumes "Steps" and not "epochs"
parser.add_argument('--scheduler_factor', type=float, default=0.5)

parser.add_argument('--flow_epochs', type=int, default=200) # was trained for 500 epochs before
parser.add_argument('--num_generated_images', type=int, default=20)

parser.add_argument('--retrain_flow_network', action='store_true', help='Train the flow model')
parser.set_defaults(retrain_flow_network=False)

parser.add_argument('--train_flow', action='store_true', help='Train the flow model')
parser.add_argument('--no-train_flow', dest='train_flow', action='store_false', help='Skip flow model training')
parser.set_defaults(train_flow=True)

parser.add_argument('--generate_images', action='store_true', help='Generate images after training')
parser.add_argument('--no-generate_images', dest='generate_images', action='store_false', help='Skip image generation')
parser.set_defaults(generate_images=True)

# NEW ARGS: VAE configuration
parser.add_argument('--use_hf_vae', action='store_true', help='Use Hugging Face VAE instead of custom VAE')
parser.add_argument('--no-use_hf_vae', dest='use_hf_vae', action='store_false', help='Use custom VAE')
parser.set_defaults(use_hf_vae=True)

parser.add_argument('--hf_vae_model', type=str, default="stabilityai/sd-vae-ft-mse", 
                    help='Hugging Face VAE model name or path') #  stabilityai/sd-vae-ft-ema, before: stabilityai/sd-vae-ft-mse

parser.add_argument('--vae_scale_factor', type=float, default=0.18215, 
                    help='Scale factor for SD VAE latents (default: 0.18215)')

parser.add_argument('--generate_vae_reconstructions', action='store_true', 
                   help='Generate VAE reconstructions to evaluate VAE quality')
parser.set_defaults(generate_vae_reconstructions=True)

parser.add_argument('--num_reconstructions', type=int, default=10,
                   help='Number of VAE reconstructions to generate')

parser.add_argument('--reaction_UNet_channels', type=int, default=100, #100 for diffusion case, 120 for non diffusion case (to compare to 18,386 mil ) 
                   help='number of channels argument to UNet that will be reaction network') #100 is default, change it back if not doing noDiffusion case for 13 mil. 105 for no diffusion, 107 for no diffusion comparison to attn case

parser.add_argument('--diffusion_UNet_channels', type=int, default=52, #32 for 13 million case, 52 for 16.759 million case
                   help='number of channels argument to UNet that will be reaction network')

parser.add_argument('--diffusion_UNet_resBlocks', type=int, default=5, 
                   help='number of resnet blocks argument to UNet that will be diffusion network')

parser.add_argument('--diffusion_UNet_attn_resolutions', type=int, nargs="+", default=(32, 16), 
                   help='specifying the resolutions at which attention is applied in the diffusion term UNet')
parser.add_argument("--diffusion_UNet_ch_mult", nargs="+", type=int,default=(1, 2), help="diffusion UNet channel multipliers")

parser.add_argument('--reaction_UNet_attn_resolutions', type=int, nargs="+", default=(32, 16), 
                   help='specifying the resolutions at which attention is applied in the diffusion term UNet')
parser.add_argument("--reaction_UNet_ch_mult", nargs="+", type=int,default=(1, 2), help="diffusion UNet channel multipliers")

parser.add_argument('--reaction_UNet_resBlocks', type=int, default=3, 
                   help='number of resnet blocks argument to UNet that will be reaction network')

# if using Attention:
parser.add_argument('--time_channels', type=int, default=64)
parser.add_argument('--time_hidden_channels', type=int, default=64)
parser.add_argument('--time_num_frequencies', type=int, default=8)

parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--reproducible', action='store_true', help='Enable reproducibility with fixed seed')
parser.add_argument('--no-reproducible', dest='reproducible', action='store_false', help='Disable reproducibility (random behavior)')
parser.set_defaults(reproducible=True)

parser.add_argument('--int_method', type=str, default='rk4', choices=['dopri5', 'rk4', 'rk2', 'euler'], help='Integration method to use')
parser.add_argument('--nsteps', type=int, default=3)

parser.add_argument('--fid_batch_size', type=int, default=32 , help='Then batch size of images to generate to reach num_img_for_FID_comp for FID computation in train loop')
parser.add_argument('--fid_patience', type=int, default=300, help='The number of epochs to wait for an FID score to beat the minimum seen so far. If no new minimum, the train loop breaks')
parser.add_argument('--temp_fid_comp_img_directory', type=str, default='/home/shahriar/FlowMatchingPredCor/CelebA/generated_images/temp')
parser.add_argument('--num_img_for_FID_comp', type=int, default=10,
                   help='Number of images to generate for FID computation every 50 epochs or so')


parser.add_argument('--penalize_acceleration', action='store_true', help='Adds term to loss term that penalizes dv/dt, or d^2x/dt^2')
parser.set_defaults(penalize_acceleration=False)
parser.add_argument('--delta_t_for_acceleration', type=float, default=0.1, 
                    help='delta_t is the time step in the denominator for the computation of dv/dt')
parser.add_argument('--lambda_acceleration', type=float, default=1.0, help='hyper-parameter that controls the weight of the acceleration penalty in the loss')


parser.add_argument('--use_ema', action='store_true', help='Use EMA for training parameters', default=False)
parser.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay factor')

parser.add_argument('--use_pretrained', action='store_true', help='Use EMA for training parameters', default=True)
parser.add_argument('--pretrained_model_path', type=str, default="/data/shahriar/models/LSUN_Bedrooms/models/PreTrainedModels/LFM_dit_LSUN_Bedrooms_256/LFM_dit_LSUN_Bedrooms_model_550.pth",
                    choices=["/data/shahriar/models/LSUN_Bedrooms/models/PreTrainedModels/LFM_dit_LSUN_Bedrooms_256/LFM_dit_LSUN_Bedrooms_model_550.pth", 
                             "/data/shahriar/models/FFHQ/HuggingFaceVAE/mse/DhariwalUNet_FromScratch/seed_0",
                             "/home/shahriar/FlowMatchingPredCor/FFHQ/models/HuggingFaceVAE/mse/DhariwalUNet_FromScratch/seed_0",
                              "/home/shahriar/FlowMatchingPredCor/FFHQ/models/PreTrainedModels/LFM_ADM_FFHQ_256/LFM_ADM_FFHQ_256_model_325.pth"
                             , "/data/shahriar/models/CelebAHQ256/models/PretrainedModels/LFM_DiT_L2_CelebA_HQ_256/DiT_L2_CelebA256_model_475.pth",
                              "/home/shahriar/FlowMatchingPredCor/LSUN_Bedrooms/models/PreTrainedModels/LFM_ADM_LSUN_Bedrooms_256/LFM_ADM_LSUN_Bedrooms_model_425.pth",
                              "/home/shahriar/FlowMatchingPredCor/LSUN_Church/models/PreTrainedModels/LFM_ADM_LSUN_Church_256/LFM_ADM_LSUN_Church_256_model_425.pth", 
                             "/home/shahriar/FlowMatchingPredCor/CelebA/models/PretrainedModels/LFM_ADM_CelebA_HQ_256/LFM_ADM_CelebA_HQ_256.pth",
                              "/home/shahriar/FlowMatchingPredCor/CelebA/models/PretrainedModels/LFM_DiT_L2_CelebA_HQ_256/DiT_L2_CelebA256_model_475.pth"], 
                    help='path to the pretrained model file')

parser.add_argument('--ffhq_pretrained_model_savepath', type=str, default="/home/shahriar/FlowMatchingPredCor/FFHQ/models/HuggingFaceVAE/mse/DhariwalUNet_FromScratch/seed_0", choices=["/home/shahriar/FlowMatchingPredCor/FFHQ/models/HuggingFaceVAE/mse/DhariwalUNet_FromScratch/seed_0", "/data/shahriar/models/FFHQ/HuggingFaceVAE/mse/DhariwalUNet_FromScratch/seed_0"])

parser.add_argument('--adm_config_path', type=str, default="/home/shahriar/FlowMatchingPredCor/networks/adm_unet_config_ffhq.yaml", choices=["/home/shahriar/FlowMatchingPredCor/networks/adm_unet_config_ffhq.yaml",
                                                                                                                                                       "/home/shahriar/FlowMatchingPredCor/networks/adm_unet_config_lsun_bedrooms.yaml",
                                                                                                                                                     "/home/shahriar/FlowMatchingPredCor/networks/adm_unet_config_lsun_church.yaml", 
                                                                                                                                                    '/home/shahriar/FlowMatchingPredCor/networks/adm_unet_config_celeba.yaml'])

parser.add_argument('--optimizer', type=str, default="AdamW", help='Whether to use AdamW or Adam to optimize during training')
parser.add_argument('--checkFID_every', type=int, default=1, help="The number of epochs to calculate FID after, during training")
parser.add_argument('--freeze_reaction_term', action='store_true', help='Freeze the reaction term parameters (base_model) so that only the diffusion term is trained.', default=False)

# Use step-based logging and FID evaluation instead of epoch-based
parser.add_argument('--use_step_logging', action='store_true',
                    help='If True, log metrics and evaluate FID every specified number of steps instead of every epoch.', default=True)
parser.add_argument('--log_every_steps', type=int, default=10,
                    help='Log wandb metrics every specified number of training steps when using step logging.')
parser.add_argument('--checkFID_every_steps', type=int, default=500,
                    help='Run FID evaluation every specified number of training steps when using step logging.')
parser.add_argument('--step_patience', type=int, default=5000,
                    help='Stop training if no FID improvement is seen for this many steps (step-based early stopping).')
# parser.add_argument('--diffusion_lr', type=float, default=5e-4,
#                     help='Learning rate for the diffusion network')
parser.add_argument('--lr', type=float, default=3e-4) #3e-4
parser.add_argument('--train_at_one_random_time_point_per_iter', action='store_true', default=False, help='Train the model at only one random time point each iteration')
parser.add_argument('--T_cosine_scheduler', type=float, default=5*60667, help='T_max for the cosine scheduler')

parser.add_argument('--use_reverse_time_convention', action='store_true', help='Use reverse time convention for training', default=True)

##############################################
parser.add_argument('--subset_frac', type=float, default=1.0, help='Fraction of dataset to use for training (e.g. 0.1 for 10%)')
########## GPS specific parser args ##########
parser.add_argument('--use_identity_graph', action='store_true', help='Freeze the reaction term parameters (base_model) so that only the diffusion term is trained for GPS2', default=False)
parser.add_argument('--diffusion_pe_dim',type=int,default=16,help='Dimensionality of projected Random-Walk PE before concatenation')
parser.add_argument('--diffusion_walk_length',type=int,default=20,help='Length of random-walk used by AuthenticRandomWalkPE')
parser.add_argument('--diffusion_gps_layers',type=int,default=16,help='Number of GPSConv layers in the GPSDiffusion block')
parser.add_argument('--gps_heads',type=int,default=4,help='Number of attention heads in each GPSConv')
parser.add_argument('--gps_dropout',type=float,default=0.0,help='Dropout rate used by GPSConv’s global attention')
parser.add_argument('--attn_type',type=str,default='multihead',choices=['multihead','performer'],help='Type of attention used by GPSConv')
parser.add_argument('--diffusion_gps_channels', type=int, default=256, #128
                    help='Number of hidden channels for the GPS diffusion term network')
##############################################

#18.386
# time_channels=32, time_hidden_channels=32, time_num_frequencies=8
# The Stable Diffusion VAE produces latent vectors with shape [4, H/8, W/8]
# When you load the VAE from stabilityai/sd-vae-ft-mse using the diffusers library, it will automatically create an encoder that outputs latent vectors of shape [batch_size, 4, height/8, width/8]


args = parser.parse_args()

# dec = torch.load(
#     os.path.join(args.model_savepath, 'hf_decoder_wrapper.pt'),
#     map_location=args.device,
#     weights_only=False  # Force full loading if needed
# )

# Set up reproducibility if requested
if args.reproducible:
    # Set all seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    
    # Define worker init function for DataLoader
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
    # Create a generator for DataLoader only
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    dataloader_kwargs = {
        'worker_init_fn': seed_worker,
        'generator': g
    }
else:
    dataloader_kwargs = {}

batch_size = args.train_batch_size
device = args.device

if args.use_wandb:
    import wandb
    if args.adj_mode=='knn':
        if args.freeze_reaction_term:
            exp_name = f"ReactionFrozen_{args.base_model}_Seed{args.seed}_{args.Diffusion_term_activation_function}_BatchSize{args.train_batch_size}_KNNAdj_k{args.knn_k}_{args.flow_model_type}"
        else:
            exp_name = f"AllTrainable_{args.base_model}_Seed{args.seed}_{args.Diffusion_term_activation_function}_BatchSize{args.train_batch_size}_KNNAdj_k{args.knn_k}_{args.flow_model_type}"
    elif args.adj_mode=='attention':
        if args.freeze_reaction_term:
            exp_name = f"ReactionFrozen_{args.base_model}_Seed{args.seed}_{args.Diffusion_term_activation_function}_BatchSize{args.train_batch_size}_AttentionAdj_{args.flow_model_type}"
        else:
            exp_name = f"AllTrainable_{args.base_model}_Seed{args.seed}_{args.Diffusion_term_activation_function}_BatchSize{args.train_batch_size}_AttentionAdj_{args.flow_model_type}"
    else:
        raise ValueError(f"Adjacency Mode not set up for experiment naming, add that in: {args.adj_mode}")
    # wandb.init(project="AFHQ-Cat-64_eldad_vae_gnn", config=vars(args), name=exp_name)
    # wandb.init(project="AFHQ_cat256_DiffReact5", config=vars(args), name=exp_name)
    # wandb.init(project="AFHQ_Cat256_nonLinearHeatDiffusion", config=vars(args), name=exp_name)
    # wandb.init(project="CelebA-HQ-256", config=vars(args), name=exp_name)
    # wandb.init(project="LSUN-Church-256", config=vars(args), name=exp_name)
    # wandb.define_metric("avg_loss_epoch", step_metric="epoch")
    wandb.init(project="LSUN-Bedrooms-256", config=vars(args), name=exp_name)
    # wandb.init(project="FFHQ-256", config=vars(args), name=exp_name)

# Create directories
os.makedirs(args.model_savepath, exist_ok=True)
os.makedirs(args.image_savepath, exist_ok=True)

# Add this to debug your VAE behavior
def diagnose_vae(enc, dec, dataloader, num_samples=5):
    """
    Diagnose VAE issues by analyzing input/output ranges and intermediate values.
    """
    print("===== VAE DIAGNOSIS =====")
    
    # Get a batch of images
    data_iter = iter(dataloader)
    x0, x1 = next(data_iter)
    x1 = x1[:num_samples].to(args.device)
    
    # Check input range
    print(f"Input images - Min: {x1.min().item():.4f}, Max: {x1.max().item():.4f}, Mean: {x1.mean().item():.4f}")
    
    # Encode - capture intermediates
    with torch.no_grad():
        # Standard encoding
        latents, log_var = enc(x1)
        print(f"Latents - Shape: {latents.shape}, Min: {latents.min().item():.4f}, Max: {latents.max().item():.4f}, Mean: {latents.mean().item():.4f}")
        
        # Decode
        reconstructions = dec(latents)
        print(f"Reconstructions - Min: {reconstructions.min().item():.4f}, Max: {reconstructions.max().item():.4f}, Mean: {reconstructions.mean().item():.4f}")
    
    return x1, latents, reconstructions


def generate_vae_reconstructions(enc, dec, dataloader, num_samples=10, save_path=args.image_savepath, is_pre_encoded=False):
    """
    Generate and save VAE reconstructions to evaluate VAE quality using torchvision.
    
    Parameters:
    -----------
    enc : torch.nn.Module or None
        VAE encoder (can be None if using pre-encoded data)
    dec : torch.nn.Module
        VAE decoder
    dataloader : torch.utils.data.DataLoader
        Dataloader for the dataset
    num_samples : int
        Number of samples to reconstruct
    save_path : str
        Path to save the reconstructions
    is_pre_encoded : bool
        Whether the dataloader contains pre-encoded data
    """
    if save_path is None:
        save_path = args.image_savepath
        
    recon_path = os.path.join(save_path, "vae_reconstructions")
    os.makedirs(recon_path, exist_ok=True)
    
    # Get a batch of images
    data_iter = iter(dataloader)
    x0, x1 = next(data_iter)
    x1 = x1[:num_samples].to(args.device)
    
    # If using pre-encoded data, x1 is already the latent vector
    if is_pre_encoded:
        print("Using pre-encoded data - skipping encoding step")
        latents = x1
        # We don't have the original images, so we'll just decode and save the results
        with torch.no_grad():
            reconstructions = dec(latents)
            reconstructions = torch.clamp(reconstructions, -1.0, 1.0)  # EXPLICIT clamping
            
        # Save reconstructions
        save_image(
            reconstructions * 0.5 + 0.5,  # Scale from [-1,1] to [0,1]
            os.path.join(recon_path, f"reconstructions_from_latents.png"),
            normalize=False  # No need to normalize again
        )
        
        print(f"Reconstructions from latents saved to {recon_path}")
        return 0  # Return a dummy MSE since we can't calculate it
    
    # Normal flow for when we have both encoder and original images
    with torch.no_grad():
        latents, log_var = enc(x1)
        reconstructions = dec(latents)
        # Add explicit clamping for metrics
        reconstructions = torch.clamp(reconstructions, -1.0, 1.0)  # EXPLICIT clamping

    # Save originals and reconstructions
    save_image(
        x1 * 0.5 + 0.5,  # Scale from [-1,1] to [0,1]
        os.path.join(recon_path, f"originals.png"),
        normalize=False  # No need to normalize again
    )
    
    save_image(
        reconstructions * 0.5 + 0.5,  # Scale from [-1,1] to [0,1]
        os.path.join(recon_path, f"reconstructions.png"),
        normalize=False  # No need to normalize again
    )
    
    # Save side-by-side comparisons
    for i in range(num_samples):
        comparison = torch.cat([x1[i].unsqueeze(0), reconstructions[i].unsqueeze(0)], dim=0)
        save_image(
            comparison * 0.5 + 0.5,  # Scale from [-1,1] to [0,1]
            os.path.join(recon_path, f"comparison_{i}.png"),
            nrow=2,
            padding=5,
            normalize=False,  # No need to normalize again
            pad_value=1      # White padding
        )
    
    # Create a grid of all comparisons
    all_originals = x1
    all_recons = reconstructions
    
    # Interleave originals and reconstructions
    rows = []
    for i in range(num_samples):
        row = torch.cat([all_originals[i].unsqueeze(0), all_recons[i].unsqueeze(0)], dim=0)
        rows.append(row)
    
    all_comparisons = torch.cat(rows, dim=0)
    
    # Save the grid image
    save_image(
        all_comparisons * 0.5 + 0.5,  # Scale from [-1,1] to [0,1]
        os.path.join(recon_path, "all_reconstructions.png"),
        nrow=2,  # 2 images per row (original and reconstruction)
        padding=5,
        normalize=False,  # No need to normalize again
        pad_value=1      # White padding
    )
    
    print(f"VAE reconstructions saved to {recon_path}")
    
    # Calculate and return reconstruction error
    mse = F.mse_loss(x1, reconstructions).item()
    print(f"Reconstruction MSE: {mse:.6f}")
    
    # Additional metrics for a more comprehensive evaluation
    # PSNR
    max_pixel_value = 2.0  # Range is [-1, 1], so max difference is 2
    psnr = 10 * torch.log10((max_pixel_value**2) / F.mse_loss(x1, reconstructions))
    print(f"Reconstruction PSNR: {psnr.item():.4f} dB")
    
    # Check if there are values outside the expected [-1, 1] range
    min_val = reconstructions.min().item()
    max_val = reconstructions.max().item()
    if min_val < -1.0 or max_val > 1.0:
        print(f"Warning: VAE produces values outside [-1, 1] range. Min: {min_val:.4f}, Max: {max_val:.4f}")
    
    return mse

# In the training section, replace the dataset and dataloader setup with:

train_loader, enc, dec = setup_training_components(args, dataloader_kwargs=dataloader_kwargs)

####################################
###### FLOW NETWORK TRAINING #######
####################################
if args.train_flow:
    # Load or create VAE
    if args.retrain_flow_network == True:
        vel_net = torch.load(os.path.join(args.model_savepath, 'vel_net_best_fid.pt'), map_location=args.device, weights_only=False)
        # vel_net = torch.load(os.path.join(args.model_savepath, 'vel_net.pt'), map_location=args.device, weights_only=False)
        # RK4, RK4step = return_RK4_functions(args.base_model)
        print("Model loaded, retraining now")
        # Freeze the reaction term if the flag is set
        if args.freeze_reaction_term:
            print("Freezing reaction term parameters.")
            for param in vel_net.reaction.parameters():
                param.requires_grad = False

        p_vel_net_trainable = sum(p.numel() for p in vel_net.parameters() if p.requires_grad)
        p_vel_net = sum(p.numel() for p in vel_net.parameters()) #all parameters, both trainable and frozen
        print(f'Number of parameters: {p_vel_net}')
        print(f'Number of trainable parameters: {p_vel_net_trainable}')
        print(f'Model type: {args.flow_model_type}')
        if args.use_wandb:
            wandb.log({"Parameters_vel_net": p_vel_net})
            wandb.log({"Total_Trainable_Parameters_vel_net": p_vel_net_trainable})
    else:
        ### CHOOSE DIFFUSION TERM ####

        if args.diffusion_network == 'resnet':
            diffusion_network = resnet(args.latent_channels, args.latent_channels, 64, num_layers=3).to(args.device)
        
        elif args.diffusion_network == 'dit':
            from networks.DiT import DiT_XS_2
            diffusion_network = DiT_XS_2(img_resolution=32, in_channels=args.latent_channels, num_classes=1, label_dropout=0.0, learn_sigma=False).to(args.device)

        elif args.diffusion_network == 'gps':
            print(f"Diffusion network: {args.diffusion_network} !!")
            from networks.GPS import GPSDiffusion, GPSDiffusion2
            diffusion_network = GPSDiffusion2(
                in_channels    = args.latent_channels,
                H              = 32,
                W              = 32,
                hidden_dim     = args.diffusion_gps_channels,
                pe_dim         = args.diffusion_pe_dim,
                walk_length    = args.diffusion_walk_length,
                num_layers     = args.diffusion_gps_layers,
                heads          = args.gps_heads,
                attn_type      = args.attn_type,
                attn_kwargs    = {'dropout': args.gps_dropout},
                time_channels  = args.time_channels,
                time_hidden    = args.time_hidden_channels,
                time_freqs     = args.time_num_frequencies,
                redraw_interval= (1000 if args.attn_type=='performer' else None),
                use_identity_graph=args.use_identity_graph
            ).to(args.device)

        elif args.diffusion_network == 'NonLinearHeat':
                
                N1_net = UNet2(
                input_channels=args.latent_channels,
                input_height=32,
                ch=args.diffusion_UNet_channels,
                output_channels=args.latent_channels,
                ch_mult=args.diffusion_UNet_ch_mult,
                num_res_blocks=args.diffusion_UNet_resBlocks, #2 #5 #3 
                attn_resolutions=args.diffusion_UNet_attn_resolutions, #(32,16) default for AFHQ-Cat and first DiT celeba-256 run
                dropout=0.0,
                resamp_with_conv=True).to(args.device)

                N2_net = UNet2(
                input_channels=args.latent_channels,
                input_height=32,
                ch=args.diffusion_UNet_channels,
                output_channels=args.latent_channels,
                ch_mult=args.diffusion_UNet_ch_mult,
                num_res_blocks=args.diffusion_UNet_resBlocks, #2 #5 #3 
                attn_resolutions=args.diffusion_UNet_attn_resolutions, #(32,16) default for AFHQ-Cat and first DiT celeba-256 run
                dropout=0.0,
                resamp_with_conv=True).to(args.device)

                NonLinearHeat_net = create_hybrid_model(
                    base_model=N1_net,
                    in_channels=args.latent_channels,
                    hidden_dim=args.diffusion_resnet_channels,
                    model_type='nonlinear_heat_eq',
                    adj_mode=args.adj_mode,
                    diffusion=args.diffusion,
                    identity_Laplacian=args.identity_Laplacian,
                    activation_type=args.Diffusion_term_activation_function,
                    diffusion_network=N2_net,
                    time_channels=args.time_channels, 
                    time_hidden_channels=args.time_hidden_channels, 
                    time_num_frequencies=args.time_num_frequencies).to(args.device) 
                diffusion_network = NonLinearHeat_net

        elif args.diffusion_network == 'None':
            diffusion_network = None

        else:
            diffusion_network = UNet2(
                        input_channels=args.latent_channels,
                        input_height=32,
                        ch=args.diffusion_UNet_channels,
                        output_channels=args.latent_channels,
                        ch_mult=args.diffusion_UNet_ch_mult,
                        num_res_blocks=args.diffusion_UNet_resBlocks, #5 #3 
                        attn_resolutions=args.diffusion_UNet_attn_resolutions, #(32,16) default for AFHQ-Cat and first DiT celeba-256 run
                        dropout=0.0,
                        resamp_with_conv=True).to(args.device)

        if args.use_pretrained == False:
            if args.base_model == 'dit':
                # Import your DiT model here
                if args.diffusion == True:
                    from networks.DiT import DiT_XS_2, DiT_B_2
                    # base_model = DiT_B_2(img_resolution=32, in_channels=args.latent_channels, num_classes=1, label_dropout=0.0, learn_sigma=False).to(args.device) #8 for previous VAE, 32 for current
                    base_model = DiT_B_2(img_resolution=32, in_channels=args.latent_channels, num_classes=1, label_dropout=0.0, learn_sigma=False).to(args.device)
                else: 
                    # load bigger model if no diffusion
                    # wondering what network 
                    from networks.DiT import DiT_XS_2_noDiffusion, DiT_M_2_noDiffusion
                    base_model = DiT_M_2_noDiffusion(img_resolution=32, in_channels=args.latent_channels, num_classes=1, label_dropout=0.0, learn_sigma=False).to(args.device)
            
            elif args.base_model == 'pnpUNet':
                base_model = UNet2(
                            input_channels=args.latent_channels,
                            input_height=32,
                            ch=args.reaction_UNet_channels,
                            output_channels=args.latent_channels,
                            ch_mult=args.reaction_UNet_ch_mult,
                            num_res_blocks=args.reaction_UNet_resBlocks, # 3 #4
                            attn_resolutions= args.reaction_UNet_attn_resolutions, #(32,16) default for AFHQ-Cat and first DiT celeba-256 run
                            dropout=0.0, #0.0, 0.1
                            resamp_with_conv=True).to(args.device)
            # elif args.base_model == 'NCSNpp':
            #     base_model = NCSNpp(get_model_config_32x32())
        else:
            if args.base_model == 'dit':
                base_model = load_pretrained_dit_model(args.pretrained_model_path, model_type="DiT-L/2", img_resolution=32, in_channels=args.latent_channels, device=args.device)
            elif args.base_model == 'adm':
                if args.dataset == 'ffhq':
                    from utils.model_utils import OutputOnlyFirst
                    base_model = torch.load(os.path.join(args.ffhq_pretrained_model_savepath, 'vel_net_best_fid.pt'), map_location=args.device, weights_only=False)
                    base_model = OutputOnlyFirst(base_model)
                else:
                    # base_model will be ADM 
                    config_path = args.adm_config_path  #'/home/shahriar/FlowMatchingPredCor/networks/adm_unet_config.yaml'
                    # # Load the model and configuration
                    if args.dataset == 'lsun_bedrooms' or args.dataset == 'ffhq':
                        untrained_unet_model, unet_config = load_edm_unet(config_path, device=args.device, use_fp16=False)
                    else:
                        untrained_unet_model, unet_config = load_adm_unet(config_path, device=args.device, use_fp16=False, use_checkpoint=False)
                        
                    unet_model = load_pretrained_unet_weights(untrained_unet_model, checkpoint_path=args.pretrained_model_path, device=args.device)
                                    # # You can now use 'unet_model' in your training loop.
                    print(f"Loaded ADM UNet with {sum(p.numel() for p in unet_model.parameters() if p.requires_grad)} parameters.")
                    base_model = unet_model 
            else:
                raise ValueError    
                   
        # Create the hybrid model using the base model
        vel_net = create_hybrid_model(
            base_model=base_model,
            in_channels=args.latent_channels,
            hidden_dim=args.diffusion_resnet_channels,
            model_type=args.flow_model_type,
            adj_mode=args.adj_mode,
            diffusion=args.diffusion,
            identity_Laplacian=args.identity_Laplacian,
            activation_type=args.Diffusion_term_activation_function,
            diffusion_network=diffusion_network,
            time_channels=args.time_channels, 
            time_hidden_channels=args.time_hidden_channels, 
            time_num_frequencies=args.time_num_frequencies).to(args.device) 

        # RK4, RK4step = return_RK4_functions(args.base_model)
        
        # Freeze the reaction term if the flag is set
        if args.freeze_reaction_term:
            print("Freezing reaction term parameters.")
            for param in vel_net.reaction.parameters():
                param.requires_grad = False

        p_vel_net = sum(p.numel() for p in vel_net.parameters()) #all parameters, both trainable and frozen
        p_vel_net_trainable = sum(p.numel() for p in vel_net.parameters() if p.requires_grad)
        print(f'Number of parameters: {p_vel_net}')
        print(f'Number of trainable parameters: {p_vel_net_trainable}')
        print(f'Model type: {args.flow_model_type}')
        if diffusion_network is not None:
            p_diffusion_net = sum(p.numel() for p in diffusion_network.parameters() if p.requires_grad)
        else:
            p_diffusion_net = 0
        if args.use_wandb:
            wandb.log({"Total_Parameters_vel_net": p_vel_net})
            wandb.log({"Total_Trainable_Parameters_vel_net": p_vel_net_trainable})
            wandb.log({"Parameters_diffusion_net": p_diffusion_net})

    
    ##############################################
    parameters = vel_net.parameters()  
    # Then, when setting up your optimizer, only parameters with requires_grad=True are included
    parameters = filter(lambda p: p.requires_grad, vel_net.parameters())
    if args.optimizer == 'AdamW':
        optimizer = optim.AdamW(parameters, lr=args.lr, betas=(0.9, 0.999))
    else:
        optimizer = optim.Adam(parameters, lr=args.lr)
    if args.use_ema:
        optimizer = EMA(optimizer, ema_decay=args.ema_decay)

    ##############################################
    # Add these variables right before the training loop starts (after scheduler setup):
    best_fid = float('inf')
    epochs_without_improvement = 0
    best_model_path = os.path.join(args.model_savepath, 'vel_net_best_fid.pt')

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.scheduler_factor, patience=args.patience, verbose=True)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.flow_epochs, eta_min=1e-5, verbose=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.T_cosine_scheduler, eta_min=5e-5, verbose=True) # for LSUN Bedrooms
    epochs = args.flow_epochs
    batches = len(train_loader)
    fid_true_vs_flow_trial=float('inf')  # initialize 
    vel_net.train()
    global_step = 0
    steps_without_improvement = 0
    early_stop = False  # We'll set this flag to True when the patience is exceeded.
    for epoch in range(epochs):
        if args.reproducible:
            # Use a different seed for each epoch by adding the epoch number
            torch.manual_seed(args.seed + epoch)
        total_loss = 0.0
        it = 0
        total_diffusion_term = 0.0
        total_reaction_term = 0.0

        for batch_idx, (x0, x1) in enumerate(train_loader):
            # If you need different random values for each batch:
            if args.reproducible:
                # You can further modify the seed with batch_idx if needed
                torch.manual_seed(args.seed + epoch * 10000 + batch_idx)

            x1 =  x1.to(args.device)
            optimizer.zero_grad()
            
            # When using pre-encoded data, x1 is already the latent vector, no need to encode
            if not args.use_pre_encoded and enc is not None:
                # VAE part - get latents from encoder (original workflow)
                with torch.no_grad():
                    latents, log_var = enc(x1)
                # Explicitly detach latents from computational graph
                latents = latents.detach()
                log_var = log_var.detach()
            else:
                # x1 is already the latent vector (pre-encoded workflow)
                latents = x1.detach()

            # Flow part (remaining code stays the same)
            if args.use_pretrained == True:
                # use convention that t=1 is noise and t=0 are the images

                if args.train_at_one_random_time_point_per_iter==True:
                    N = 1   # for example, split into 1,2 or 3, or N pieces
                    B = latents.shape[0]
                    # compute a list of counts summing to B,
                    # distributing any remainder into the first few chunks
                    base = B // N
                    remainder = B % N
                    sizes = [(base + 1 if i < remainder else base) for i in range(N)]
                    # e.g. B=12, N=5 → sizes=[3,3,2,2,2]
                    # sample N independent scalars
                    r = torch.rand(N, device=args.device)   # shape [N]
                    # expand r into a length-B tensor, repeating each scalar for its chunk
                    t = r.repeat_interleave(torch.tensor(sizes, device=args.device))  # shape [B]
                else:
                    t = torch.rand(latents.shape[0], device=args.device)

                noise = torch.randn_like(latents)
                zt = (1 - t.view(-1, 1, 1, 1)) * latents + (1e-5 + (1 - 1e-5) * t.view(-1, 1, 1, 1)) * noise    #(1 - t.view(-1, 1, 1, 1)) * latents + t.view(-1, 1, 1, 1) * noise
                vf = ( (1 - 1e-5) * noise ) - latents
            else:
                # use convention that t=0 is noise and t=1 are the images
                if args.train_at_one_random_time_point_per_iter==True:
                    N = 1   # for example, split into 1,2 or 3, or N pieces
                    B = latents.shape[0]
                    # compute a list of counts summing to B,
                    # distributing any remainder into the first few chunks
                    base = B // N
                    remainder = B % N
                    sizes = [(base + 1 if i < remainder else base) for i in range(N)]
                    # e.g. B=12, N=5 → sizes=[3,3,2,2,2]
                    # sample N independent scalars
                    r = torch.rand(N, device=args.device)   # shape [N]
                    # expand r into a length-B tensor, repeating each scalar for its chunk
                    t = r.repeat_interleave(torch.tensor(sizes, device=args.device))  # shape [B]
                else:
                    t = torch.rand(latents.shape[0], device=args.device)

                noise = torch.randn_like(latents)
                # zt = t.view(-1, 1, 1, 1) * latents + (1 - t.view(-1, 1, 1, 1)) * noise
                # vf = latents - noise
                vf = ( (1 - 1e-5) * latents ) - noise
                zt = (1e-5 + (1 - 1e-5) * t.view(-1, 1, 1, 1)) * latents + (1 - t.view(-1, 1, 1, 1)) * noise

            vc, diffusion_term, reaction_term = vel_net(zt, t)  


            # vc, diffusion_term, reaction_term = vel_net(x=zt, t=t) #the use of keyword args is important because 'dit' requires (t,x) and not (x,t), which is opposite of other models
                
            with torch.no_grad():
                nrmv = F.mse_loss(vf, torch.zeros_like(vf))
            
            velocity_loss = F.mse_loss(vf, vc) / nrmv

            # with torch.autograd.detect_anomaly():
            loss = velocity_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(vel_net.parameters(), max_norm=1.0) # clip gradients to prevent exploding gradients
            
            ################# CHECK THAT NO GRADIENTS ARE FLOWING THROUGH VAE ########
            # Verify no gradients in VAE (add this check)
            if epoch == 0 and it == 0:  # Only check in the first iteration to avoid slowdown
                has_grad = False
                
                # Check encoder if it exists (won't exist when using pre-encoded data)
                if enc is not None:
                    for name, param in enc.named_parameters():
                        if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
                            print(f"WARNING: Gradient detected in VAE encoder parameter: {name}")
                            has_grad = True
                
                # Check decoder (should always exist)
                for name, param in dec.named_parameters():
                    if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
                        print(f"WARNING: Gradient detected in VAE decoder parameter: {name}")
                        has_grad = True
                
                if not has_grad:
                    print("✓ Verified: No gradients flowing through VAE parameters")
            ###################

            ###################
            optimizer.step()
            
            scheduler.step()  # if doing this every iteration, have an appropriate number of "steps" in the scheduler
            # if args.use_step_logging:
                # scheduler.step(loss)

            total_loss += loss.item()
            total_diffusion_term += diffusion_term.item()
            total_reaction_term += reaction_term.item()

            print('%d  %d/%d  %3.2e' % (epoch, it, batches, loss))
            it += 1
            # Increment the global step counter for each batch processed.
            global_step += 1

            # If step-based logging is enabled, log at the specified step frequency.
            if args.use_step_logging:
                if global_step % args.log_every_steps == 0 and args.use_wandb:
                    current_lr = optimizer.param_groups[0]['lr']

                    wandb.log({
                        "global_step": global_step,
                        "batch_loss": loss.item(),
                        "diffusion_term.abs().max()":diffusion_term.item(),
                        "reaction_term.abs().max()":reaction_term.item(),
                        "lr": current_lr 
                    })

                # If it’s time to run FID evaluation based on steps:
                if global_step % args.checkFID_every_steps == 0:
                    # Optionally clear cache and set seed before evaluation.
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    if args.reproducible:
                        torch.manual_seed(args.seed + 1000000 + global_step)

                    if args.use_ema:
                        optimizer.swap_parameters_with_ema(store_params_in_ema=True)
                    vel_net.eval()
                    with torch.no_grad():
                        if args.use_pretrained or args.use_reverse_time_convention:
                            time_convention = 'reverse'
                        else:
                            time_convention = 'standard'
                        gen_img_save_dir, _ = generate_flow_matched_images(
                            vel_net, decoder=dec,
                            num_samples=args.num_img_for_FID_comp,
                            batch_size=args.fid_batch_size,
                            save_dir=args.temp_fid_comp_img_directory,
                            device=args.device,
                            latent_channels=args.latent_channels,
                            latent_size=32,
                            nsteps=args.nsteps,
                            base_model=args.base_model,
                            int_method=args.int_method,
                            time_convention=time_convention,
                            return_NFE=False
                        )
                        torch.cuda.empty_cache()
                        gc.collect()
                        fid_true_vs_flow_trial = fid.compute_fid(
                            args.temp_fid_comp_img_directory,
                            dataset_name=args.cleanfid_dataset_name,
                            dataset_split="custom",
                            mode="clean",
                            num_workers=2,
                            batch_size=2,
                            device=torch.device(args.device) 
                        )

                        del gen_img_save_dir
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # # --- regularly checkpoint model + optimizer ---
                    # ckpt = {
                    #     'step': global_step,
                    #     'fid': fid_true_vs_flow_trial,
                    #     'model_state_dict': vel_net.state_dict(),
                    #     'optimizer_state_dict': optimizer.state_dict(),
                    # }
                    # torch.save(ckpt, os.path.join(args.model_savepath,
                    #                               f'checkpoint_step_{global_step}.pth'))
                    torch.save(vel_net, os.path.join(args.model_savepath, 'vel_net.pt'))
                    # ----------------------------------------------------------

                    # Check if the new FID score is the best so far.
                    if fid_true_vs_flow_trial < best_fid:
                        best_fid = fid_true_vs_flow_trial
                        torch.save(vel_net, best_model_path)
                        print(f"New best FID: {best_fid:.4f} at step {global_step}, model saved.")
                        steps_without_improvement = 0
                    else:
                        steps_without_improvement += args.checkFID_every_steps
                        print(f"No FID improvement for {steps_without_improvement} steps.")
                    # Log the FID score.
                    if args.use_wandb:
                        wandb.log({"global_step": global_step, "FID": fid_true_vs_flow_trial})


                    vel_net.train()
                    if args.use_ema:
                        optimizer.swap_parameters_with_ema(store_params_in_ema=True)
                    
                    torch.cuda.empty_cache()
                    gc.collect()
                    # Check for early stopping (step-based).
                    if steps_without_improvement >= args.step_patience:
                        print(f"Early stopping triggered: No FID improvement for {steps_without_improvement} steps.")
                        early_stop = True
                        break  # Break out of the inner (batch) loop.

            # Optionally, if early stopping was triggered, exit the inner loop.
            if early_stop:
                break
        # At the end of the epoch, check if you need to break out completely.
        if early_stop:
            break

        if not args.use_step_logging:
            if epoch % args.checkFID_every == 0:
                # Clear memory
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                # Set seed before generating images for FID
                if args.reproducible:
                    torch.manual_seed(args.seed + 1000000 + epoch)  # Use a large offset
                
                if args.use_ema:
                    optimizer.swap_parameters_with_ema(store_params_in_ema=True)
                vel_net.eval()
                with torch.no_grad():
                    # compute FID
                    if args.use_pretrained:
                        time_convention = 'reverse'
                    else:
                        time_convention = 'standard'
                    gen_img_save_dir, _ = generate_flow_matched_images(vel_net, decoder=dec, num_samples=args.num_img_for_FID_comp, batch_size=args.fid_batch_size, 
                                    save_dir=args.temp_fid_comp_img_directory, device=args.device, 
                                    latent_channels=args.latent_channels, latent_size=32, nsteps=args.nsteps, 
                                    base_model=args.base_model, int_method=args.int_method, time_convention=time_convention, return_NFE=False)
                    
                    fid_true_vs_flow_trial = fid.compute_fid(
                        args.temp_fid_comp_img_directory,
                        dataset_name=args.cleanfid_dataset_name,
                        dataset_split="custom",
                        mode="clean",
                        num_workers=16
                    )

                torch.cuda.empty_cache()    # Clear CUDA cache
                gc.collect()                # Force garbage collection

                # # --- regularly checkpoint model + optimizer ---
                # ckpt = {
                #     'epoch': epoch,
                #     'fid': fid_true_vs_flow_trial,
                #     'model_state_dict': vel_net.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                # }
                # torch.save(ckpt, os.path.join(args.model_savepath,
                #                               f'checkpoint_epoch_{epoch}.pth'))
                torch.save(vel_net, os.path.join(args.model_savepath, 'vel_net.pt'))
                # ----------------------------------------------------------

                # Check if this is the best FID score
                if fid_true_vs_flow_trial < best_fid:
                    best_fid = fid_true_vs_flow_trial
                    # Save the best model
                    torch.save(vel_net, best_model_path)
                    print(f"New best FID: {best_fid:.4f}, saved model to {best_model_path}")
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += args.checkFID_every  # Since we check every "args.checkFID_every" epochs
                
                # After FID computation but before the second EMA swap
                # fid_true_vs_flow_trial  # Explicitly delete variables
                torch.cuda.empty_cache()    # Clear CUDA cache
                gc.collect()                # Force garbage collection

                vel_net.train()
                if args.use_ema:
                    optimizer.swap_parameters_with_ema(store_params_in_ema=True)
                # Early stopping check
                if epochs_without_improvement >= args.fid_patience:
                    # fid_patience was kept to 300
                    print(f"Early stopping after {epoch+1} epochs. No FID improvement for 300 epochs.")
                    break
        
        avg_total_loss = total_loss / batches
        avg_diffusion_term = total_diffusion_term / batches
        avg_reaction_term = total_reaction_term / batches

        # scheduler.step() #scheduler.step(avg_total_loss)

        # if not args.use_step_logging:
        #     scheduler.step(avg_total_loss)
        print(f"======= Flow Epoch {epoch+1}/{epochs}, Loss: {avg_total_loss:.4f}")
        
        current_lr = optimizer.param_groups[0]['lr']
        # Log flow loss to wandb
        if args.use_wandb:
            wandb.log({
                "flow_epoch": epoch+1,
                "flow_total_loss": avg_total_loss,
                "log10(flow_total_loss)": math.log10(avg_total_loss),
                "Average diffusion_term.abs().max() per batch": avg_diffusion_term,
                "Average reaction_term.abs().max() per batch": avg_reaction_term,
                "log10[Average_diffusion_term.abs().max()_per_batch]": math.log10(avg_diffusion_term),             
                "log10[Average_reaction_term.abs().max()_per_batch]": math.log10(avg_reaction_term),
                f"FID ({args.num_img_for_FID_comp} gen. images, {args.int_method}, {args.nsteps} steps)": fid_true_vs_flow_trial,
                "lr": current_lr
            })
    
        # Save the trained model directly to the specified save path

    # Save the trained model directly to the specified save path
    if args.use_ema:
        optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        
    torch.save(vel_net, os.path.join(args.model_savepath, 'vel_net.pt'))
    print(f"Trained velocity network saved to: {os.path.join(args.model_savepath, 'vel_net.pt')}")

   


# # -----------------------------
# # Image Generation with Best FID Model
# # -----------------------------
# if args.generate_images:
#     if args.use_pretrained:
#         time_convention = 'reverse'
#     else:
#         time_convention = 'standard'

#     # Load the VAE decoder
#     if args.use_hf_vae:
#         try:
#             dec = torch.load(os.path.join(args.model_savepath, 'hf_decoder_wrapper.pt'), map_location=args.device, weights_only=False)
#             print("Loaded saved HF VAE decoder wrapper")
#         except FileNotFoundError:
#             print("Creating new HF VAE decoder wrapper")
#             _, dec = create_hf_vae_wrappers(pretrained_model_name_or_path=args.hf_vae_model, device=args.device)
#     else:
#         dec = torch.load(os.path.join(args.model_savepath, 'decoder.pt'), map_location=args.device, weights_only=False)
    
#     # Load the best model by FID score
#     best_model_path = os.path.join(args.model_savepath, 'vel_net_best_fid.pt')
#     if os.path.exists(best_model_path):
#         vel_net = torch.load(best_model_path, map_location=args.device, weights_only=False)
#         print(f"Using best FID model from {best_model_path}")
#     else:
#         vel_net = torch.load(os.path.join(args.model_savepath, 'vel_net.pt'), map_location=args.device, weights_only=False)
#         print("Using final model (no best FID model found)")
    
#     # Set models to evaluation mode
#     vel_net.eval()
#     dec.eval()
    
#     # Create a subfolder for generated images
#     save_subfolder = os.path.join(args.image_savepath, f"{args.flow_model_type}")
#     os.makedirs(save_subfolder, exist_ok=True)
    
#     # Generate and save images
#     gen_img_save_dir, _ =  generate_flow_matched_images(
#         vel_net, decoder=dec, 
#         num_samples=args.num_generated_images, 
#         batch_size=16,  # Smaller batch for higher quality generation
#         save_dir=save_subfolder, 
#         device=args.device,
#         latent_channels=args.latent_channels, 
#         latent_size=32, 
#         nsteps=args.nsteps,
#         base_model=args.base_model, 
#         int_method=args.int_method,
#         time_convention=time_convention,
#         return_NFE=False
#         )
    
#     print(f"Generated images saved to: {save_subfolder}")
    
#     # Create PDF grid of all generated images
#     model_name = f"{args.flow_model_type}_{args.base_model}"
#     pdf_path = os.path.join(save_subfolder, f"{model_name}_grid.pdf")
#     plot_images_grid(save_subfolder, pdf_path)
#     print(f"Created PDF grid of generated images at: {pdf_path}")
    
#     # Generate VAE reconstructions if requested
#     if args.generate_vae_reconstructions:
#         print("Generating VAE reconstructions...")
#         mse = generate_vae_reconstructions(
#             enc if 'enc' in locals() else None,
#             dec, 
#             train_loader,
#             num_samples=args.num_reconstructions,
#             is_pre_encoded=args.use_pre_encoded
#         )
#         if args.use_wandb:
#             wandb.log({"vae_reconstruction_mse": mse})


if args.use_wandb:
    wandb.finish()