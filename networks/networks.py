import torch
import torch.nn as nn
from diffusers import AutoencoderKL

########## HUGGING FACE pretrained model wrappers #############
class EncoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        
    def forward(self, x):
        # Our dataloader already outputs images in range [-1, 1]
        # No need to normalize, SD VAE already expects [-1, 1]
        
        # Get latent representation - this returns the mean and log variance
        posterior = self.vae.encode(x).latent_dist
        latents = posterior.mean
        log_var = posterior.logvar
        
        return latents * self.vae.config.scaling_factor, log_var
    
    def reparameterize(self, mean, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

class DecoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        
    def forward(self, latents):
        # Scale the latents as expected by SD VAE
        latents = latents / self.vae.config.scaling_factor
        
        # Decode the latent representation to image space
        with torch.no_grad():
            # Output will be in range [-1, 1] which matches our dataloader
            decoded = self.vae.decode(latents).sample
            
            # # Add explicit clamping here
            # decoded = torch.clamp(decoded, -1.0, 1.0)
            
        return decoded
    

class DecoderWrapper_withGradients(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        
    def forward(self, latents):
        # Scale the latents as expected by SD VAE
        latents = latents / self.vae.config.scaling_factor
        
        # Decode the latent representation to image space

        # Output will be in range [-1, 1] which matches our dataloader
        decoded = self.vae.decode(latents).sample
            
        return decoded

def create_hf_vae_wrappers(pretrained_model_name_or_path="stabilityai/sd-vae-ft-mse", device="cuda:0", decoder_require_gradients=False):
    """
    Create encoder and decoder wrapper classes for the Hugging Face VAE
    to match the interface expected by the existing code.
    """
    # Load the pretrained VAE
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path)
    # Print config information to debug
    print(f"VAE config summary:")
    print(f"- Scaling factor: {getattr(vae.config, 'scaling_factor', 0.18215)}")
    print(f"- Sample size: {getattr(vae.config, 'sample_size', None)}")
    print(f"- In channels: {vae.config.in_channels}")
    print(f"- Out channels: {vae.config.out_channels}")
    print(f"- Latent channels: {vae.config.latent_channels}")
    vae.to(device)
    vae.eval()  # Set to evaluation mode
    
    # Explicitly freeze all VAE parameters to ensure they're not trained
    for param in vae.parameters():
        param.requires_grad = False
    
    # Create wrapper instances
    encoder_wrapper = EncoderWrapper(vae).to(device)

    if not decoder_require_gradients:
        decoder_wrapper = DecoderWrapper(vae).to(device)
    else:
        decoder_wrapper = DecoderWrapper_withGradients(vae).to(device)
    
    # Double-check and ensure all parameters in wrappers have requires_grad=False
    for param in encoder_wrapper.parameters():
        param.requires_grad = False
    
    for param in decoder_wrapper.parameters():
        param.requires_grad = False
    
    return encoder_wrapper, decoder_wrapper


