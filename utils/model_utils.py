# model_utils.py

from omegaconf import OmegaConf
from networks.ADM.adm_unet import UNetModel
import torch

def load_adm_unet(config_path: str, device: str = 'cuda', use_fp16: bool = False, use_checkpoint: bool = False):
    """
    Loads the ADM UNet configuration from a YAML file and instantiates the UNet model.

    Args:
        config_path (str): Path to the YAML configuration file.
        device (str): The device on which to place the model (default: 'cuda').
        use_fp16 (bool): Whether to use FP16 precision (default: False).
        use_checkpoint (bool): Whether to enable gradient checkpointing (default: False).

    Returns:
        model (UNetModel): The instantiated UNet model.
        config (OmegaConf): The loaded configuration.
    """
    # Load the configuration
    config = OmegaConf.load(config_path)
    
    # Instantiate the UNet. Note that the training code divides image_size by 8 internally.
    model = UNetModel(
        image_size=config.image_size // 8,  # Converts full resolution (e.g., 256) to the internal resolution (e.g., 32)
        in_channels=config.num_in_channels,
        model_channels=config.nf,
        out_channels=config.num_out_channels,
        num_res_blocks=config.num_res_blocks,
        attention_resolutions=config.attn_resolutions,
        dropout=config.dropout,
        channel_mult=config.ch_mult,
        conv_resample=config.resamp_with_conv,
        dims=2,
        num_classes=config.num_classes,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=config.num_heads,
        num_head_channels=config.num_head_channels,
        num_heads_upsample=config.num_heads_upsample,
        use_scale_shift_norm=config.use_scale_shift_norm,
        resblock_updown=config.resblock_updown,
        use_new_attention_order=config.use_new_attention_order,
    )
    
    # Move model to the specified device
    model.to(device)
    return model, config



from omegaconf import OmegaConf
from networks.ADM.adm_unet import UNetModelAttn
import torch

def load_adm_unet_attention(config_path: str, device: str = 'cuda', use_fp16: bool = False, use_checkpoint: bool = False):
    """
    Loads the ADM UNet with attention configuration from a YAML file and instantiates the model.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        device (str): The device on which to place the model (default: 'cuda').
        use_fp16 (bool): Whether to use FP16 precision (default: False).
        use_checkpoint (bool): Whether to enable gradient checkpointing (default: False).
        
    Returns:
        model (UNetModelAttn): The instantiated UNet model with attention.
        config (OmegaConf): The loaded configuration.
    """
    # Load the configuration
    config = OmegaConf.load(config_path)
    
    # Instantiate the UNet with attention
    model = UNetModelAttn(
        image_size=config.image_size // 8,  # Converts full resolution to internal resolution
        in_channels=config.num_in_channels,
        model_channels=config.nf,
        out_channels=config.num_out_channels,
        num_res_blocks=config.num_res_blocks,
        attention_resolutions=config.attn_resolutions,
        dropout=config.dropout,
        channel_mult=config.ch_mult,
        conv_resample=config.resamp_with_conv,
        dims=2,
        num_classes=config.num_classes,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=config.num_heads,
        num_head_channels=config.num_head_channels,
        num_heads_upsample=config.num_heads_upsample,
        use_scale_shift_norm=config.use_scale_shift_norm,
        resblock_updown=config.resblock_updown,
        use_new_attention_order=config.use_new_attention_order,
        use_spatial_transformer=config.use_spatial_transformer,
        transformer_depth=config.transformer_depth,
        context_dim=config.context_dim,
        legacy=config.legacy
    )
    
    # Move model to the specified device
    model.to(device)
    
    return model, config



from omegaconf import OmegaConf
import torch
from networks.ADM.EDM import DhariwalUNet  # Make sure this import is correct for your project structure

def load_edm_unet(config_path: str, device: str = 'cuda', use_fp16: bool = False):
    """
    Loads the EDM UNet configuration from a YAML file and instantiates the UNet model.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        device (str): The device on which to place the model (default: 'cuda').
        use_fp16 (bool): Whether to use FP16 precision (default: False).
        
    Returns:
        model (DhariwalUNet): The instantiated UNet model.
        config (OmegaConf): The loaded configuration.
    """
    # Load the configuration
    config = OmegaConf.load(config_path)
    
    # Instantiate the UNet
    model = DhariwalUNet(
        img_resolution=config.image_size // config.f,
        in_channels=config.num_in_channels,
        out_channels=config.num_out_channels,
        label_dim=config.label_dim,
        augment_dim=0,
        model_channels=config.nf,
        channel_mult=config.ch_mult,
        channel_mult_emb=4,
        num_blocks=config.num_res_blocks,
        attn_resolutions=config.attn_resolutions,
        dropout=config.dropout,
        label_dropout=config.label_dropout
    )
    
    # Move model to the specified device
    model.to(device)
    
    # Convert to fp16 if requested
    if use_fp16:
        model = model.half()
    
    return model, config



import torch
import torch.nn as nn

class OutputOnlyFirst(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)[0]
