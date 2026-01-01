import os
import sys
# Add the parent directory of folder1 to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.eldad_networks import ASPP


class TimeEmbedding(nn.Module):
    def __init__(self, n_channels, hidden_dim=32, num_freqs=8):
        super().__init__()
        self.n_channels = n_channels
        self.num_freqs = num_freqs

        # Positional encoding (learned frequencies)
        self.freqs = nn.Parameter(torch.randn(num_freqs))  # Trainable frequencies
        self.fc1 = nn.Linear(2 * num_freqs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_channels)
        self.activation = nn.SiLU()  # Smooth activation

    def forward(self, t):
        # Positional encoding
        t = t.view(-1, 1)  # Ensure it's a column vector
        sinusoids = torch.cat([torch.sin(2 * torch.pi * self.freqs * t), 
                               torch.cos(2 * torch.pi * self.freqs * t)], dim=-1)
        
        # Pass through MLP
        x = self.activation(self.fc1(sinusoids))
        x = self.fc2(x)
        return x  # Output shape: (batch_size, n_channels)

    
# class AdjacencyGenerator(nn.Module):
#     """
#     Generates an adjacency matrix for the batch items.
#     Can be a learnable module or a fixed function.
#     """
#     def __init__(self, mode='knn', learnable=True, temperature=1.0):
#         super(AdjacencyGenerator, self).__init__()
#         self.mode = mode
#         self.learnable = learnable
#         # self.temperature = nn.Parameter(torch.tensor(temperature)) if learnable else temperature
#         # Fix: Properly handle when temperature is already a tensor
#         if isinstance(temperature, torch.Tensor):
#             self.temperature = nn.Parameter(temperature.clone().detach()) if learnable else temperature
#         else:
#             self.temperature = nn.Parameter(torch.tensor(temperature)) if learnable else temperature
        
#     def forward(self, x):
#         # x shape: [batch_size, feature_dim, h, w]
#         batch_size = x.shape[0]
        
#         # Flatten and normalize feature vectors for similarity computation
#         x_flat = x.view(batch_size, -1)  # [batch_size, feature_dim*h*w]
#         x_norm = F.normalize(x_flat, p=2, dim=1)
        
#         if self.mode == 'cosine':
#             # Compute cosine similarity
#             adj = torch.mm(x_norm, x_norm.t())  # [batch_size, batch_size]
#             # Apply temperature scaling
#             adj = adj / self.temperature
#             # Apply softmax to create row-normalized adjacency
#             adj = F.softmax(adj, dim=1)
#         elif self.mode == 'gaussian':
#             # Compute pairwise distances
#             dists = torch.cdist(x_flat, x_flat, p=2)
#             # Convert to similarity with Gaussian kernel
#             adj = torch.exp(-dists / (2.0 * self.temperature * self.temperature)) 
#         elif self.mode == 'knn':
#             # Compute cosine similarity
#             adj = torch.mm(x_norm, x_norm.t())
#             # Keep top-k connections per node (k=3 by default)
#             k = min(3, batch_size - 1)
#             topk, indices = torch.topk(adj, k=k+1, dim=1)
#             mask = torch.zeros_like(adj)
#             mask.scatter_(1, indices, 1)
#             adj = adj * mask
#             # Normalize
#             row_sum = adj.sum(dim=1, keepdim=True) + 1e-6
#             adj = adj / row_sum
            
#         else:
#             # Default: identity matrix (no graph effects)
#             adj = torch.eye(batch_size, device=x.device)
            
#         return adj


class AdjacencyGenerator(nn.Module):
    """
    Generates an adjacency matrix for the batch items.
    Can be a learnable module or a fixed function.
    Supports different modes: 'cosine', 'gaussian', 'knn', 'identity'
    """
    def __init__(self, mode='gaussian', learnable=True, temperature=1.0, k=20):
        # k = 5 before
        super(AdjacencyGenerator, self).__init__()
        self.mode = mode
        self.learnable = learnable
        self.k=k
        # Properly handle when temperature is already a tensor
        if isinstance(temperature, torch.Tensor):
            self.temperature = nn.Parameter(temperature.clone().detach()) if learnable else temperature
        else:
            self.temperature = nn.Parameter(torch.tensor(temperature)) if learnable else temperature
        
        if self.mode == 'knn':
            # Initialize learnable edge weights for top-k neighbors
            # Create learnable edge weights
            self.edge_weights = nn.Parameter(torch.empty(k + 1))
            
            # Initialize close to 1.0 with small random perturbations
            nn.init.normal_(self.edge_weights, mean=1.0, std=0.01)
            
    def forward(self, x):
        """
        Generate adjacency matrix based on the input tensor.
        
        Args:
            x: Input tensor of shape [B, C, H, W] or [B, D]
            
        Returns:
            Adjacency matrix of shape [B, B]
        """
        # x shape: [batch_size, feature_dim, h, w] or [batch_size, feature_dim]
        batch_size = x.shape[0]

        # Flatten if needed
        if len(x.shape) > 2:
            x_flat = x.view(batch_size, -1)
        else:
            x_flat = x
        
        if self.mode == 'knn':
            if batch_size<(self.k+1):
                k_temp = batch_size-1
            else:
                k_temp = self.k
            x_norm = F.normalize(x_flat, p=2, dim=1)
            adj = torch.mm(x_norm, x_norm.t())  # similarity matrix
            
            topk, indices = torch.topk(adj, k= k_temp + 1, dim=1)  # includes self-loop
            
            mask = torch.zeros_like(adj)
            mask.scatter_(1, indices, 1)
            
            # Assign learnable edge weights instead of binary connections
            weighted_adj = torch.zeros_like(adj)
            for neighbor_rank in range(k_temp + 1):
                neighbor_indices = indices[:, neighbor_rank]
                weighted_adj[torch.arange(batch_size), neighbor_indices] = self.edge_weights[neighbor_rank]
            
            # Apply masking to maintain KNN structure
            adj = weighted_adj * mask
            
            # Normalize adjacency
            row_sum = adj.sum(dim=1, keepdim=True) + 1e-8
            adj = adj / row_sum
        
        elif self.mode == 'cosine':
            x_norm = F.normalize(x_flat, p=2, dim=1)
            adj = torch.mm(x_norm, x_norm.t()) / self.temperature
            adj = F.softmax(adj, dim=1)

        elif self.mode == 'gaussian':
            norm_squared = torch.sum(x_flat**2, dim=1, keepdim=True)
            dist_squared = norm_squared + norm_squared.t() - 2 * torch.mm(x_flat, x_flat.t())
            adj = torch.exp(-dist_squared / (2.0 * self.temperature**2 + 1e-8))

        else:
            adj = torch.eye(batch_size, device=x.device)

        return adj

    
    # CURRENTLY RUNNING !!!


# class AdjGenerator_Attention(nn.Module):
#     '''
#     Generates an adjacency matrix for the batch items.
#     Can be a learnable module or a fixed function.
#     '''
#     def __init__(self, mode='attention', learnable=True, temperature=1.0):
#         super(AdjGenerator_Attention, self).__init__()
#         self.mode = mode
#         self.attn = torch.nn.MultiheadAttention(
#                 4*32*32,
#                 num_heads=1,
#                 batch_first=True)
#     def forward(self, x):
#         # x shape: [batch_size, c, h, w]
#         batch_size = x.shape[0]
#         # Flatten and normalize feature vectors for similarity computation
#         x_flat = x.view(batch_size, -1)  # [batch_size, c*h*w]
#         x_flat = x_flat.unsqueeze(0) # makes it [1, batch_size, c*h*w]

#         _, learned_adj = self.attn(x_flat, x_flat, x_flat) # h is like H = MLP(learned_adj @ x_flat) 
        
#         # Remove the singleton dimension
#         learned_adj = learned_adj.squeeze(0)  # [batch_size, batch_size]

#         return learned_adj

class AttentionWithOutProj(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # Create in-projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # # No out-projection is defined
        
    def forward(self, query, key, value=None):
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        # Compute scaled dot-product attention weights
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        # Directly use the weighted sum as the output
        output = torch.matmul(attn_weights, V)
        # output = None 
        return output, attn_weights
    
class AdjGenerator_Attention(nn.Module):
    '''
    Generates an adjacency matrix for the batch items.
    Uses convolutional operations to reduce spatial dimensions before computing attention.
    '''
    def __init__(self, mode='attention'):
        super(AdjGenerator_Attention, self).__init__()
        self.mode = mode
        # Parallel dilated convolutions for multi-scale feature extraction
        self.dilation1 = nn.Conv2d(4, 1, kernel_size=3, padding=1, dilation=1)
        self.dilation2 = nn.Conv2d(4, 1, kernel_size=3, padding=2, dilation=2)
        self.dilation4 = nn.Conv2d(4, 1, kernel_size=3, padding=4, dilation=4)
        self.dilation8 = nn.Conv2d(4, 1, kernel_size=3, padding=8, dilation=8)
        # ReLU activation
        self.relu = nn.ReLU()
        # Pooling layer for downsampling
        self.pool = nn.AvgPool2d(4, 4)  # From 32x32 to 8x8
        # Calculate new embedding dimension
        reduced_dim = 4 * 8 * 8  # 4 channels from parallel dilations, 8x8 spatial size
        # Attention mechanism
        self.attn = AttentionNoOutProj(reduced_dim, num_heads=1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        # Apply parallel dilated convolutions
        feat1 = self.relu(self.dilation1(x))
        feat2 = self.relu(self.dilation2(x))
        feat4 = self.relu(self.dilation4(x))
        feat8 = self.relu(self.dilation8(x))
        # Concatenate features along channel dimension
        multi_scale_features = torch.cat([feat1, feat2, feat4, feat8], dim=1)  # [B, 4, 32, 32]
        # Downsample spatial dimensions
        x_reduced = self.pool(multi_scale_features)  # [B, 4, 8, 8]
        # Flatten and prepare for attention
        x_flat = x_reduced.view(batch_size, -1)
        x_flat = x_flat.unsqueeze(0)
        # Compute attention
        _, learned_adj = self.attn(x_flat, x_flat, x_flat)
        learned_adj = learned_adj.squeeze(0)
        return learned_adj


class AdjGenerator_Attention_ASPP_Time(nn.Module):
    '''
    Generates an adjacency matrix for the batch items.
    Uses ASPP block to reduce spatial dimensions before computing attention.
    Incorporates time vector into the feature representation.
    '''
    def __init__(self, in_channels=4, out_channels=4, mode='attention', n_time_channels=None):
        super(AdjGenerator_Attention_ASPP_Time, self).__init__()
        self.mode = mode
        
        # ASPP block for multi-scale feature extraction
        self.aspp = ASPP(in_channels, out_channels)
        
        # Downsampling layer (from 32x32 to 8x8)
        self.downsample = nn.AvgPool2d(4, 4)
        
        # Get flattened dimension for spatial features
        spatial_feat_dim = out_channels * 8 * 8  # 4 channels, 8x8 spatial size
        
        # Linear projection for time features (if provided)
        self.time_proj = None
        if n_time_channels is not None:
            self.time_proj = nn.Linear(n_time_channels, 64)  # Project time to fixed dimension
            total_feat_dim = spatial_feat_dim + 64
        else:
            total_feat_dim = spatial_feat_dim
        
        # Attention mechanism
        self.attn = AttentionWithOutProj(total_feat_dim, num_heads=2)
    
    def forward(self, x, t=None):
        batch_size = x.shape[0]
        
        # Apply ASPP block
        x = self.aspp(x)
        
        # Downsample to 8x8
        x_reduced = self.downsample(x)  # [B, 4, 8, 8]

        x_flat = x_reduced.view(batch_size, -1)  # [B, 256]
        
        # Incorporate time features if provided
        if t is not None and self.time_proj is not None:
            t_proj = self.time_proj(t)  # [B, 64]
            x_flat = torch.cat([x_flat, t_proj], dim=1)  # [B, 256+64]
        
    
        x_flat = x_flat.unsqueeze(0)  # [1, B, feat_dim]
        
        _, learned_adj = self.attn(x_flat, x_flat, x_flat)
        learned_adj = learned_adj.squeeze(0)  # [B, B]
        A_sym = (learned_adj + learned_adj.transpose(-1, -2)) / 2
        epsilon = 1e-4
        A_stable = A_sym + epsilon * torch.eye(batch_size, device=A_sym.device)

        return A_stable
    
def nodeGrad(x, adj):
    """
    Compute node gradients for batch items as nodes with dense adjacency matrix.
    For each node i, computes weighted differences with all other nodes j.
    
    Args:
        x: Node features of shape [B, C, H, W] or [B, D]
        adj: Adjacency matrix of shape [B, B]
        
    Returns:
        Gradient tensor of shape [B, B, D] or [B, B, C, H, W]
    """
    batch_size = x.shape[0]
    
    # If x is 4D tensor [B, C, H, W], flatten to [B, C*H*W]
    original_shape = x.shape
    if len(x.shape) > 2:
        x_flat = x.view(batch_size, -1)
    else:
        x_flat = x
    
    # Compute all pairwise differences efficiently
    # For each node i, compute x_i - x_j for all j
    x_i = x_flat.unsqueeze(1)  # [B, 1, D]
    x_j = x_flat.unsqueeze(0)  # [1, B, D]
    diff = x_i - x_j  # [B, B, D]
    
    # Apply adjacency weights
    adj_expanded = adj.unsqueeze(-1)  # [B, B, 1]
    weighted_diff = diff * adj_expanded  # [B, B, D]
    
    # Reshape back to original format if needed
    if len(original_shape) > 2:
        C, H, W = original_shape[1], original_shape[2], original_shape[3]
        return weighted_diff.view(batch_size, batch_size, C, H, W)
    
    return weighted_diff


def edgeDiv(grad_tensor, adj):
    """
    Compute edge divergence for dense adjacency matrix.
    For each node i, aggregates the weighted gradients from all connected nodes.
    
    Args:
        grad_tensor: Gradient tensor from nodeGrad, shape [B, B, D] or [B, B, C, H, W]
        adj: Adjacency matrix of shape [B, B]
        
    Returns:
        Divergence tensor of shape [B, D] or [B, C, H, W]
    """
    # For 5D tensor [B, B, C, H, W], flatten last dimensions
    if len(grad_tensor.shape) > 3:
        batch_size = grad_tensor.shape[0]
        C, H, W = grad_tensor.shape[2], grad_tensor.shape[3], grad_tensor.shape[4]
        grad_flat = grad_tensor.view(batch_size, batch_size, -1)
    else:
        grad_flat = grad_tensor
        batch_size = grad_flat.shape[0]
    
    # Sum over all neighbors (dimension 1)
    div = torch.sum(grad_flat, dim=1)  # [B, D]
    
    # Reshape back to original format if needed
    if len(grad_tensor.shape) > 3:
        return div.view(batch_size, C, H, W)
    
    return div
    
import torch
import torch.nn.functional as F

def compute_graph_laplacian(x, adj_generator, laplacian_type='normalized'):
    """
    Compute the graph Laplacian matrix for batch data.
    
    Args:
        x: Input tensor of shape [batch_size, feature_dim, h, w]
        within adj_generator, adj_mode: The mode for computing adjacency ('cosine', 'gaussian', 'knn')
        laplacian_type: Type of Laplacian to compute ('normalized', 'random_walk', 'combinatorial')
        temperature is set inside adj_generator so its use here is just so the fid script works
        
    Returns:
        L: Graph Laplacian matrix of shape [batch_size, batch_size]
    """
    batch_size = x.shape[0]
    
    # Create adjacency generator with specified settings
    adj_gen = adj_generator #AdjacencyGenerator(mode=adj_mode, learnable=True, temperature=temperature)
    
    # Get adjacency matrix
    A = adj_gen(x)
    
    # Compute degree matrix (sum of each row of adjacency matrix)
    D = torch.sum(A, dim=1)
    D_mat = torch.diag(D)
    
    if laplacian_type == 'combinatorial':
        # L = D - A (Combinatorial Laplacian)
        L = D_mat - A
    
    elif laplacian_type == 'random_walk':
        # L = I - D^(-1) * A (Random Walk Laplacian)
        D_inv = torch.diag(1.0 / (D + 1e-8))
        L = torch.eye(batch_size, device=x.device) - torch.mm(D_inv, A)
    
    else:  # normalized (default)
        # L = I - D^(-1/2) * A * D^(-1/2) (Normalized Laplacian)
        D_inv_sqrt = torch.diag(torch.pow(D + 1e-8, -0.5))
        L = torch.eye(batch_size, device=x.device) - torch.mm(torch.mm(D_inv_sqrt, A), D_inv_sqrt)
    
    return L

def compute_graph_laplacian_with_timeEmbed(x, t, adj_generator, laplacian_type='normalized'):
    """
    Compute the graph Laplacian matrix for batch data.
    
    Args:
        x: Input tensor of shape [batch_size, feature_dim, h, w]
        within adj_generator, adj_mode: The mode for computing adjacency ('cosine', 'gaussian', 'knn')
        laplacian_type: Type of Laplacian to compute ('normalized', 'random_walk', 'combinatorial')
        temperature is set inside adj_generator so its use here is just so the fid script works
        
    Returns:
        L: Graph Laplacian matrix of shape [batch_size, batch_size]
    """
    batch_size = x.shape[0]
    
    # Create adjacency generator with specified settings
    adj_gen = adj_generator #AdjacencyGenerator(mode=adj_mode, learnable=True, temperature=temperature)
    
    # Get adjacency matrix
    A = adj_gen(x, t) #only works with attention
    
    # Compute degree matrix (sum of each row of adjacency matrix)
    D = torch.sum(A, dim=1)
    D_mat = torch.diag(D)
    
    if laplacian_type == 'combinatorial':
        # L = D - A (Combinatorial Laplacian)
        L = D_mat - A
    
    elif laplacian_type == 'random_walk':
        # L = I - D^(-1) * A (Random Walk Laplacian)
        D_inv = torch.diag(1.0 / (D + 1e-8))
        L = torch.eye(batch_size, device=x.device) - torch.mm(D_inv, A)
    
    else:  # normalized (default)
        # L = I - D^(-1/2) * A * D^(-1/2) (Normalized Laplacian)
        D_inv_sqrt = torch.diag(torch.pow(D + 1e-8, -0.5))
        L = torch.eye(batch_size, device=x.device) - torch.mm(torch.mm(D_inv_sqrt, A), D_inv_sqrt)
    
    return L
# def compute_graph_laplacian(x, laplacian_type='normalized', temperature=1.0, adj_mode='gaussian'):
#     """
#     For FID computation for models that were trained using this style.

#     Compute the graph Laplacian matrix for batch data.
    
#     Args:
#         x: Input tensor of shape [batch_size, feature_dim, h, w]
#         within adj_generator, adj_mode: The mode for computing adjacency ('cosine', 'gaussian', 'knn')
#         laplacian_type: Type of Laplacian to compute ('normalized', 'random_walk', 'combinatorial')
#         temperature is set inside adj_generator so its use here is just so the fid script works
        
#     Returns:
#         L: Graph Laplacian matrix of shape [batch_size, batch_size]
#     """
#     batch_size = x.shape[0]
    
#     # Create adjacency generator with specified settings
#     adj_gen = AdjacencyGenerator(mode=adj_mode, learnable=True, temperature=temperature)
    
#     # Get adjacency matrix
#     A = adj_gen(x)
    
#     # Compute degree matrix (sum of each row of adjacency matrix)
#     D = torch.sum(A, dim=1)
#     D_mat = torch.diag(D)
    
#     if laplacian_type == 'combinatorial':
#         # L = D - A (Combinatorial Laplacian)
#         L = D_mat - A
    
#     elif laplacian_type == 'random_walk':
#         # L = I - D^(-1) * A (Random Walk Laplacian)
#         D_inv = torch.diag(1.0 / (D + 1e-8))
#         L = torch.eye(batch_size, device=x.device) - torch.mm(D_inv, A)
    
#     else:  # normalized (default)
#         # L = I - D^(-1/2) * A * D^(-1/2) (Normalized Laplacian)
#         D_inv_sqrt = torch.diag(torch.pow(D + 1e-8, -0.5))
#         L = torch.eye(batch_size, device=x.device) - torch.mm(torch.mm(D_inv_sqrt, A), D_inv_sqrt)
    
#     return L

    
from networks.eldad_networks import resnet, Encoder, Decoder, ASPP
class ReactionDiffusion2(nn.Module):
    """
        # Setup the system
        # dX/dt = L(k,t)@X + f(X,t)      # if f(X,t) = grad phi(X,t)  <=> min 0.5*x^L*x + phi(x,t)
        # dx1/dt = k1*L*x1 + f1(x1,x2,t)
        # dx2/dt = k2*L*x2 + f2(x1,x2,t)
    """
    def __init__(self, 
                 base_flow_network,
                 in_channels, 
                 adj_mode='gaussian',
                 learnable_adj=True,
                 use_nonlinearity=True,
                 diffusion=True
                 ):
        super(ReactionDiffusion2, self).__init__()
        
        # Base flow network (DiT, ResNet, etc.) - produces velocity field Z0
        self.reaction = base_flow_network
        self.diffusionCoef = resnet(in_channels, in_channels, 128, num_layers=3) #num_layers=3
        self.diffusion = diffusion # whether to use diffusion term or not
        self.adj_generator = AdjacencyGenerator(mode=adj_mode, learnable=learnable_adj)
        
        self.use_nonlinearity = use_nonlinearity
        if use_nonlinearity:
            self.activation = nn.ELU() #nn.Softmax(dim=2)  #nn.ELU() #nn.SiLU()  # Smooth activation function 
        
        self.closing_layer = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x, t):
        # change the above to (self, x, t) if not using DiT as the base model
        # Z0 = DiT(X) - base flow computation that produces velocity field
        # Shape: [batch_size, channels, h, w]
        z0 = self.reaction(x, t) # if not using DiT
        # z0 = self.reaction(t, x) # if using DiT

        # with torch.no_grad(): #NOTE : !!
        #     # Compute graph Laplacian based on current adjacency mode
        #     L = compute_graph_laplacian(x, adj_mode=self.adj_generator.mode, 
        #                             temperature=self.adj_generator.temperature)
        
        ####################################################################
        if self.diffusion==True:
            # # Compute graph Laplacian based on current adjacency mode
            # L = compute_graph_laplacian(x, self.adj_generator)
            
            ####################################### TEST with no Laplacian #############
            L = torch.eye(x.shape[0], x.shape[0], device=x.device)
            ######################################################
            # # Reshape for graph operation
            batch_size, channels, h, w = x.shape
            x_flat = x.view(batch_size, -1)  # [batch_size, channels*h*w]
            
            # kappa = self.activation(self.diffusionCoef(x, t))
            # kappa = self.diffusionCoef(x, t)  # try: no activation

            #### NEW: for each batch item and for each channel, apply activation separately on the 8x8 patches
            # Reshape to [B, 16, 64] where 64 = 8×8
            kappa_temp = self.diffusionCoef(x, t).view(batch_size, channels, -1)
            # Apply activation (for example, softmax) over the last dimension (spatial)
            kappa = self.activation(kappa_temp).view(batch_size, channels, h, w)


            diffusion_term = kappa.view(batch_size,-1) * torch.mm(L, x_flat) 
            reaction_term = z0.view(batch_size, -1)
            z1_flat = diffusion_term + reaction_term # plot contributions curve

            z1 = z1_flat.view(batch_size, channels, h, w)

            return z1, diffusion_term.abs().max(), reaction_term.abs().max()
            
        else:

            return z0, 1e-5+torch.zeros(1,1), 1e-5+torch.zeros(1,1) 


class ReactionDiffusion(nn.Module):
    # use this for previous models that didn't take in diffusion argument in the constructor (needed for example for the fid script)
    """
        # Setup the system
        # dX/dt = L(k,t)@X + f(X,t)      # if f(X,t) = grad phi(X,t)  <=> min 0.5*x^L*x + phi(x,t)
        # dx1/dt = k1*L*x1 + f1(x1,x2,t)
        # dx2/dt = k2*L*x2 + f2(x1,x2,t)
    """
    def __init__(self, 
                 base_flow_network,
                 in_channels, 
                 adj_mode='cosine',
                 learnable_adj=True,
                 use_nonlinearity=True):
        super(ReactionDiffusion, self).__init__()
        
        # Base flow network (DiT, ResNet, etc.) - produces velocity field Z0
        self.reaction = base_flow_network
        self.diffusionCoef = resnet(in_channels, in_channels, 128, num_layers=3) #num_layers=3
        self.adj_generator = AdjacencyGenerator(mode=adj_mode, learnable=learnable_adj)
        
        self.use_nonlinearity = use_nonlinearity
        if use_nonlinearity:
            self.activation = nn.ELU() #nn.Softmax(dim=2)  #nn.ELU() #nn.Softmax()  #nn.SiLU()  # Smooth activation function 
        
        self.closing_layer = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x, t):
        # change the above to (self, x, t) if not using DiT as the base model
        # Z0 = DiT(X) - base flow computation that produces velocity field
        # Shape: [batch_size, channels, h, w]
        z0 = self.reaction(x, t) # if not using DiT
        # z0 = self.reaction(t, x) # if using DiT

        # with torch.no_grad(): #NOTE : !!
        #     # Compute graph Laplacian based on current adjacency mode
        #     L = compute_graph_laplacian(x, adj_mode=self.adj_generator.mode, 
        #                             temperature=self.adj_generator.temperature)
        
        ####################################################################
        # Compute graph Laplacian based on current adjacency mode
        L = compute_graph_laplacian(x, adj_mode=self.adj_generator.mode, 
                                    temperature=self.adj_generator.temperature)
        # # Reshape for graph operation
        batch_size, channels, h, w = x.shape
        x_flat = x.view(batch_size, -1)  # [batch_size, channels*h*w]
        
        # kappa = self.activation(self.diffusionCoef(x, t))
        # kappa = self.diffusionCoef(x, t)  # try: no activation

        #### NEW: for each batch item and for each channel, apply activation separately on the 8x8 patches
        # Reshape to [B, 16, 64] where 64 = 8×8
        kappa_temp = self.diffusionCoef(x, t).view(batch_size, channels, -1)
        # Apply activation (for example, softmax) over the last dimension (spatial)
        kappa = self.activation(kappa_temp).view(batch_size, channels, h, w)


        diffusion_term = kappa.view(batch_size,-1) * torch.mm(L, x_flat) 
        reaction_term = z0.view(batch_size, -1)
        z1_flat = diffusion_term + reaction_term # plot contributions curve

        z1 = z1_flat.view(batch_size, channels, h, w)

        return z1, diffusion_term.abs().max(), reaction_term.abs().max()
            


class ReactionDiffusion3(nn.Module):
    """
        # Setup the system
        # dX/dt = L(k,t)@X + f(X,t)      # if f(X,t) = grad phi(X,t)  <=> min 0.5*x^L*x + phi(x,t)
        # dx1/dt = k1*L*x1 + f1(x1,x2,t)
        # dx2/dt = k2*L*x2 + f2(x1,x2,t)
    """
    def __init__(self, 
                base_flow_network,
                in_channels, 
                adj_mode='gaussian',
                learnable_adj=True,
                diffusion=True,
                identity_Laplacian=False,
                activation_type='ELU',  # New parameter for activation function
                diffusion_resnet_channels=64
                ):
        super(ReactionDiffusion3, self).__init__()
        
        # Base flow network (DiT, ResNet, etc.) - produces velocity field Z0
        self.reaction = base_flow_network
        
        self.diffusion = diffusion # whether to use diffusion term or not
        self.identity_Laplacian = identity_Laplacian

        if not self.identity_Laplacian: #edit , was if self.identity_Laplacian
            self.adj_generator = AdjacencyGenerator(mode=adj_mode, learnable=learnable_adj)
        
        if self.diffusion==True:
            self.diffusionCoef = resnet(in_channels, in_channels, diffusion_resnet_channels, num_layers=3) #128, num_layers=3


        if activation_type == 'Softmax':
            self.activation = nn.Softmax(dim=2)
        elif activation_type == 'SiLU':
            self.activation = nn.SiLU()
        else:  # Default is ELU
            self.activation = nn.ELU()
        
        self.closing_layer = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x, t):
        # change the above to (self, x, t) if not using DiT as the base model
        # Z0 = DiT(X) - base flow computation that produces velocity field
        # Shape: [batch_size, channels, h, w]
        z0 = self.reaction(x, t) # if not using DiT
        # z0 = self.reaction(t, x) # if using DiT

        ####################################################################
        if self.diffusion==True:
            if self.identity_Laplacian:
                ######################### TEST with no Laplacian #############
                L = torch.eye(x.shape[0], x.shape[0], device=x.device)
                ######################################################
            else:
                # Compute graph Laplacian based on current adjacency mode
                L = compute_graph_laplacian(x, self.adj_generator)
            
            # # Reshape for graph operation
            batch_size, channels, h, w = x.shape
            x_flat = x.view(batch_size, -1)  # [batch_size, channels*h*w]
            
            # kappa = self.activation(self.diffusionCoef(x, t))
            # kappa = self.diffusionCoef(x, t)  # try: no activation

            #### NEW: for each batch item and for each channel, apply activation separately on the 8x8 patches
            # For example, reshape to [B, C, H*W] 
            kappa_temp = self.diffusionCoef(x, t).view(batch_size, channels, -1)
            # Apply activation (for example, softmax) over the last dimension (spatial)
            kappa = self.activation(kappa_temp).view(batch_size, channels, h, w)


            diffusion_term = kappa.view(batch_size,-1) * torch.mm(L, x_flat) 
            reaction_term = z0.view(batch_size, -1)
            z1_flat = diffusion_term + reaction_term # plot contributions curve

            z1 = z1_flat.view(batch_size, channels, h, w)

            return z1, diffusion_term.abs().max(), reaction_term.abs().max()
            
        else:

            return z0, 1e-6+torch.zeros(1,1), z0.abs().max()



class ReactionDiffusion4(nn.Module):
    """
        # Setup the system
        # dX/dt = L(k,t)@X + f(X,t)      # if f(X,t) = grad phi(X,t)  <=> min 0.5*x^L*x + phi(x,t)
        # dx1/dt = k1*L*x1 + f1(x1,x2,t)
        # dx2/dt = k2*L*x2 + f2(x1,x2,t)
    """
    def __init__(self, 
                base_flow_network,
                in_channels, 
                adj_mode='gaussian',
                learnable_adj=True,
                diffusion=True,
                identity_Laplacian=False,
                activation_type='ELU',  # New parameter for activation function
                diffusion_network = resnet 
                ):
        super(ReactionDiffusion4, self).__init__()
        
        # Base flow network (DiT, ResNet, etc.) - produces velocity field Z0
        self.reaction = base_flow_network
        
        self.diffusion = diffusion # whether to use diffusion term or not
        self.identity_Laplacian = identity_Laplacian

        if not self.identity_Laplacian: #edit , was if self.identity_Laplacian
            self.adj_generator = AdjacencyGenerator(mode=adj_mode, learnable=learnable_adj)
        
        if self.diffusion==True:
            self.diffusionCoef = diffusion_network #resnet(in_channels, in_channels, diffusion_resnet_channels, num_layers=3) #128, num_layers=3


        if activation_type == 'Softmax':
            self.activation = nn.Softmax(dim=2)
        elif activation_type == 'SiLU':
            self.activation = nn.SiLU()
        else:  # Default is ELU
            self.activation = nn.ELU()
        
        self.closing_layer = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x, t):
        # change the above to (self, x, t) if not using DiT as the base model
        # Z0 = DiT(X) - base flow computation that produces velocity field
        # Shape: [batch_size, channels, h, w]
        z0 = self.reaction(x, t) # if not using DiT
        # z0 = self.reaction(t, x) # if using DiT

        ####################################################################
        if self.diffusion==True:
            if self.identity_Laplacian:
                ######################### TEST with no Laplacian #############
                L = torch.eye(x.shape[0], x.shape[0], device=x.device)
                ######################################################
            else:
                # Compute graph Laplacian based on current adjacency mode
                L = compute_graph_laplacian(x, self.adj_generator)
            
            # # Reshape for graph operation
            batch_size, channels, h, w = x.shape
            x_flat = x.view(batch_size, -1)  # [batch_size, channels*h*w]
            
            # kappa = self.activation(self.diffusionCoef(x, t))
            # kappa = self.diffusionCoef(x, t)  # try: no activation

            #### NEW: for each batch item and for each channel, apply activation separately on the 8x8 patches
            # For example, reshape to [B, C, H*W] 
            kappa_temp = self.diffusionCoef(x, t).view(batch_size, channels, -1)
            # Apply activation (for example, softmax) over the last dimension (spatial)
            kappa = self.activation(kappa_temp).view(batch_size, channels, h, w)


            diffusion_term = kappa.view(batch_size,-1) * torch.mm(L, x_flat) 
            reaction_term = z0.view(batch_size, -1)
            z1_flat = diffusion_term + reaction_term # plot contributions curve

            z1 = z1_flat.view(batch_size, channels, h, w)

            return z1, diffusion_term.abs().max(), reaction_term.abs().max()
            
        else:

            return z0, 1e-6+torch.zeros(1,1), z0.abs().max()
        

class ReactionDiffusion5(nn.Module):
    """
        # Setup the system
        # dX/dt = L(k,t)@X + f(X,t)      # if f(X,t) = grad phi(X,t)  <=> min 0.5*x^L*x + phi(x,t)
        # dx1/dt = k1*L*x1 + f1(x1,x2,t)
        # dx2/dt = k2*L*x2 + f2(x1,x2,t)
    """
    def __init__(self, 
                base_flow_network,
                in_channels, 
                adj_mode='gaussian',
                learnable_adj=True,
                diffusion=True,
                identity_Laplacian=False,
                activation_type='ELU',  # New parameter for activation function
                diffusion_network = resnet 
                ):
        super(ReactionDiffusion5, self).__init__()
        
        # Base flow network (DiT, ResNet, etc.) - produces velocity field Z0
        self.reaction = base_flow_network
        
        self.diffusion = diffusion # whether to use diffusion term or not
        self.identity_Laplacian = identity_Laplacian

        if not self.identity_Laplacian: #edit , was if self.identity_Laplacian
            self.adj_generator = AdjacencyGenerator(mode=adj_mode, learnable=learnable_adj)
        
        if self.diffusion==True:
            self.diffusionCoef = diffusion_network #resnet(in_channels, in_channels, diffusion_resnet_channels, num_layers=3) #128, num_layers=3


        if activation_type == 'Softmax':
            self.activation = nn.Softmax(dim=1)  # Now over channels instead of spatial dimensions
        elif activation_type == 'SiLU':
            self.activation = nn.SiLU()
        elif activation_type == 'ReLU':
            self.activation = nn.ReLU()
        else:  # Default is ELU
            self.activation = nn.ELU()
        
        self.closing_layer = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x, t):
        # change the above to (self, x, t) if not using DiT as the base model
        # Z0 = DiT(X) - base flow computation that produces velocity field
        # Shape: [batch_size, channels, h, w]
        z0 = self.reaction(x, t) # if not using DiT
        # z0 = self.reaction(t, x) # if using DiT

        ####################################################################
        if self.diffusion==True:
            if self.identity_Laplacian:
                ######################### TEST with no Laplacian #############
                L = torch.eye(x.shape[0], x.shape[0], device=x.device)
                ######################################################
            else:
                # Compute graph Laplacian based on current adjacency mode
                L = compute_graph_laplacian(x, self.adj_generator)
            
            # # Reshape for graph operation
            batch_size, channels, h, w = x.shape
            x_flat = x.view(batch_size, -1)  # [batch_size, channels*h*w]
            
            # kappa = self.activation(self.diffusionCoef(x, t))
            # kappa = self.diffusionCoef(x, t)  # try: no activation

            # Modified implementation (one coefficient per channel per batch):
            kappa_temp = self.diffusionCoef(x, t)
            # Global average pooling to get one value per channel
            kappa_temp = F.adaptive_avg_pool2d(kappa_temp, (1, 1))  # Shape: [batch_size, channels, 1, 1]
            # Apply activation to these channel-wise coefficients
            kappa = self.activation(kappa_temp.view(batch_size, channels))  # Shape: [batch_size, channels]
            # Expand to match spatial dimensions for multiplication
            kappa_expanded = kappa.view(batch_size, channels, 1, 1).expand(-1, -1, h, w)
            # Apply diffusion
            # Alternative version
            kappa_expanded_flat = kappa_expanded.reshape(batch_size, -1)
            diffusion_term = kappa_expanded_flat * torch.mm(L, x_flat)

            reaction_term = z0.view(batch_size, -1)
            z1_flat = diffusion_term + reaction_term # plot contributions curve

            z1 = z1_flat.view(batch_size, channels, h, w)

            return z1, diffusion_term.abs().max(), reaction_term.abs().max()
            
        else:

            return z0, 1e-6+torch.zeros(1,1), z0.abs().max()
        

class ReactionDiffusion6(nn.Module):
    """
        # Setup the system
        # dX/dt = L(k,t)@X + f(X,t)      # if f(X,t) = grad phi(X,t)  <=> min 0.5*x^L*x + phi(x,t)
        # dx1/dt = k1*L*x1 + f1(x1,x2,t)
        # dx2/dt = k2*L*x2 + f2(x1,x2,t)
    """
    def __init__(self, 
                base_flow_network,
                in_channels, 
                adj_mode='gaussian',
                learnable_adj=True,
                diffusion=True,
                identity_Laplacian=False,
                activation_type='ELU',  # New parameter for activation function
                diffusion_network = resnet 
                ):
        super(ReactionDiffusion6, self).__init__()
        
        # Base flow network (DiT, ResNet, etc.) - produces velocity field Z0
        self.reaction = base_flow_network
        
        self.diffusion = diffusion # whether to use diffusion term or not
        self.identity_Laplacian = identity_Laplacian

        if not self.identity_Laplacian: #edit , was if self.identity_Laplacian
            if self.diffusion==True:
                if adj_mode == 'attention':
                    self.adj_generator = AdjGenerator_Attention(mode=adj_mode, learnable=learnable_adj)
                    
                else:
                    self.adj_generator = AdjacencyGenerator(mode=adj_mode, learnable=learnable_adj)
        
        if self.diffusion==True:
            self.diffusionCoef = diffusion_network #resnet(in_channels, in_channels, diffusion_resnet_channels, num_layers=3) #128, num_layers=3


        if activation_type == 'Softmax':
            self.activation = nn.Softmax(dim=1)  # Now over channels instead of spatial dimensions
        elif activation_type == 'SiLU':
            self.activation = nn.SiLU()
        elif activation_type == 'ReLU':
            self.activation = nn.ReLU()
        else:  # Default is ELU
            self.activation = nn.ELU()
        
        self.closing_layer = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x, t):
        # change the above to (self, x, t) if not using DiT as the base model
        # Z0 = DiT(X) - base flow computation that produces velocity field
        # Shape: [batch_size, channels, h, w]

        # z0 = self.reaction(x, t) # if not using DiT
        z0 = self.reaction(t, x) # if using DiT

        ####################################################################
        if self.diffusion==True:
            if self.identity_Laplacian:
                ######################### TEST with no Laplacian #############
                L = torch.eye(x.shape[0], x.shape[0], device=x.device)
                ######################################################
            else:
                # Compute graph Laplacian based on current adjacency mode
                L = compute_graph_laplacian(x, self.adj_generator)
            
            # # Reshape for graph operation
            batch_size, channels, h, w = x.shape
            x_flat = x.view(batch_size, -1)  # [batch_size, channels*h*w]
            
            # kappa = self.activation(self.diffusionCoef(x, t))
            # kappa = self.diffusionCoef(x, t)  # try: no activation

            # Modified implementation (one coefficient per channel per batch):
            kappa_temp = self.diffusionCoef(x, t) 
            # Global average pooling to get one value per channel
            kappa_temp = F.adaptive_avg_pool2d(kappa_temp, (1, 1))  # Shape: [batch_size, channels, 1, 1]
            # Apply activation to these channel-wise coefficients
            kappa = self.activation(kappa_temp.view(batch_size, channels))  # Shape: [batch_size, channels]
            # Expand to match spatial dimensions for multiplication
            kappa_expanded = kappa.view(batch_size, channels, 1, 1).expand(-1, -1, h, w)
            # Apply diffusion
            # Alternative version
            kappa_expanded_flat = kappa_expanded.reshape(batch_size, -1)
            diffusion_term = kappa_expanded_flat * torch.mm(L, x_flat)

            reaction_term = z0.view(batch_size, -1)
            z1_flat = diffusion_term + reaction_term # plot contributions curve

            z1 = z1_flat.view(batch_size, channels, h, w)

            return z1, diffusion_term.abs().max(), reaction_term.abs().max()
            
        else:

            return z0, 1e-6+torch.zeros(1,1), z0.abs().max()

class ReactionDiffusion6_improved(nn.Module):
    """
        # Setup the system
        # dX/dt = L(k,t)@X + f(X,t)      # if f(X,t) = grad phi(X,t)  <=> min 0.5*x^L*x + phi(x,t)
        # dx1/dt = k1*L*x1 + f1(x1,x2,t)
        # dx2/dt = k2*L*x2 + f2(x1,x2,t)
    """
    def __init__(self, 
                base_flow_network,
                in_channels, 
                adj_mode='knn',
                learnable_adj=True,
                diffusion=True,
                identity_Laplacian=False,
                activation_type='ELU',  # New parameter for activation function
                diffusion_network = resnet,
                knn_k = 20
                ):
        super(ReactionDiffusion6_improved, self).__init__()
        
        # Base flow network (DiT, ResNet, etc.) - produces velocity field Z0
        self.reaction = base_flow_network
        
        self.diffusion = diffusion # whether to use diffusion term or not
        self.identity_Laplacian = identity_Laplacian

        if not self.identity_Laplacian: #edit , was if self.identity_Laplacian
            if self.diffusion==True:
                if adj_mode == 'attention':
                    self.adj_generator = AdjGenerator_Attention(mode=adj_mode, learnable=learnable_adj)
                    
                else:
                    self.adj_generator = AdjacencyGenerator(mode=adj_mode, learnable=learnable_adj, k=knn_k)
        
        if self.diffusion==True:
            self.diffusionCoef = diffusion_network #resnet(in_channels, in_channels, diffusion_resnet_channels, num_layers=3) #128, num_layers=3


        if activation_type == 'Softmax':
            self.activation = nn.Softmax(dim=1)  # Now over channels instead of spatial dimensions
        elif activation_type == 'SiLU':
            self.activation = nn.SiLU()
        elif activation_type == 'ReLU':
            self.activation = nn.ReLU()
        else:  # Default is ELU
            self.activation = nn.ELU()
        
        self.closing_layer = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x, t):
        # change the above to (self, x, t) if not using DiT as the base model
        # Z0 = DiT(X) - base flow computation that produces velocity field
        # Shape: [batch_size, channels, h, w]

        if self.reaction.__class__.__name__ == 'DiT':
            z0 = self.reaction(t, x) # if using DiT
        else:
            z0 = self.reaction(x, t) # if not using DiT
        

        ####################################################################
        if self.diffusion==True:
            if self.identity_Laplacian:
                ######################### TEST with no Laplacian #############
                L = torch.eye(x.shape[0], x.shape[0], device=x.device)
                ######################################################
            else:
                # Compute graph Laplacian based on current adjacency mode
                L = compute_graph_laplacian(x, self.adj_generator)
            
            # # Reshape for graph operation
            batch_size, channels, h, w = x.shape
            x_flat = x.view(batch_size, -1)  # [batch_size, channels*h*w]
            
            # kappa = self.activation(self.diffusionCoef(x, t))
            # kappa = self.diffusionCoef(x, t)  # try: no activation

            # Modified implementation (one coefficient per channel per batch):
            if self.diffusionCoef.__class__.__name__ == 'DiT':
                kappa_temp = self.diffusionCoef(t, x) 
            else:
                kappa_temp = self.diffusionCoef(x, t) 
            # Global average pooling to get one value per channel
            kappa_temp = F.adaptive_avg_pool2d(kappa_temp, (1, 1))  # Shape: [batch_size, channels, 1, 1]
            # Apply activation to these channel-wise coefficients
            kappa = self.activation(kappa_temp.view(batch_size, channels))  # Shape: [batch_size, channels]
            # Expand to match spatial dimensions for multiplication
            kappa_expanded = kappa.view(batch_size, channels, 1, 1).expand(-1, -1, h, w)
            # Apply diffusion
            # Alternative version
            kappa_expanded_flat = kappa_expanded.reshape(batch_size, -1)
            diffusion_term = kappa_expanded_flat * torch.mm(L, x_flat)

            reaction_term = z0.view(batch_size, -1)
            z1_flat = diffusion_term + reaction_term # plot contributions curve

            z1 = z1_flat.view(batch_size, channels, h, w)

            return z1, diffusion_term.abs().max(), reaction_term.abs().max()
            
        else:

            return z0, 1e-6+torch.zeros(1,1), z0.abs().max()
        

class ReactionDiffusion7(nn.Module):
    """
        # Setup the system
        # dX/dt = L(k,t)@X + f(X,t)      # if f(X,t) = grad phi(X,t)  <=> min 0.5*x^L*x + phi(x,t)
        # dx1/dt = k1*L*x1 + f1(x1,x2,t)
        # dx2/dt = k2*L*x2 + f2(x1,x2,t)
    """
    def __init__(self, 
                base_flow_network,
                in_channels=4, 
                adj_mode='attention',
                learnable_adj=True,
                diffusion=True,
                identity_Laplacian=False,
                activation_type='ELU',  # New parameter for activation function
                diffusion_network = resnet,
                time_channels = 32,
                time_hidden_channels=32,
                time_num_frequencies=8 
                ):
        super(ReactionDiffusion7, self).__init__()
        
        # this network calls a laplacian function that only works with the attention mode of neighbour finding
        # Base flow network (DiT, ResNet, etc.) - produces velocity field Z0
        self.reaction = base_flow_network
        
        self.diffusion = diffusion # whether to use diffusion term or not
        self.identity_Laplacian = identity_Laplacian

        if not self.identity_Laplacian: #edit , was if self.identity_Laplacian
            if self.diffusion:
                if adj_mode == 'attention':
                    self.adj_generator = AdjGenerator_Attention_ASPP_Time(in_channels=in_channels, out_channels=in_channels, mode=adj_mode,
                                                                         n_time_channels=time_channels)  #AdjGenerator_Attention(mode=adj_mode, learnable=learnable_adj)
                    
                else:
                    self.adj_generator = AdjacencyGenerator(mode=adj_mode, learnable=learnable_adj)
        
        if self.diffusion==True:
            self.time_embed = TimeEmbedding(n_channels=time_channels, hidden_dim=time_hidden_channels, num_freqs=time_num_frequencies)
            self.diffusionCoef = diffusion_network #resnet(in_channels, in_channels, diffusion_resnet_channels, num_layers=3) #128, num_layers=3


        if activation_type == 'Softmax':
            self.activation = nn.Softmax(dim=1)  # Now over channels instead of spatial dimensions
        elif activation_type == 'SiLU':
            self.activation = nn.SiLU()
        elif activation_type == 'ReLU':
            self.activation = nn.ReLU()
        else:  # Default is ELU
            self.activation = nn.ELU()
        
    def forward(self, x, t):
        # change the above to (self, x, t) if not using DiT as the base model
        # Z0 = DiT(X) - base flow computation that produces velocity field
        # Shape: [batch_size, channels, h, w]
        # z0 = self.reaction(x, t) # if not using DiT
        # z0 = self.reaction(t, x) # if using DiT

        if self.reaction.__class__.__name__ == 'DiT':
            z0 = self.reaction(t, x) # if using DiT
        else:
            z0 = self.reaction(x, t) # if not using DiT
        ####################################################################
        if self.diffusion==True:
            if self.identity_Laplacian:
                ######################### TEST with no Laplacian #############
                L = torch.eye(x.shape[0], x.shape[0], device=x.device)
                ######################################################
            else:
                t_embedded = self.time_embed(t)
                # Compute graph Laplacian based on current adjacency mode
                L = compute_graph_laplacian_with_timeEmbed(x, t_embedded, self.adj_generator)
                # L = compute_graph_laplacian(x, self.adj_generator)
            
            # # Reshape for graph operation
            batch_size, channels, h, w = x.shape
            x_flat = x.view(batch_size, -1)  # [batch_size, channels*h*w]
            
            # kappa = self.activation(self.diffusionCoef(x, t))
            # kappa = self.diffusionCoef(x, t)  # try: no activation

            # Modified implementation (one coefficient per channel per batch):
            if self.diffusionCoef.__class__.__name__ == 'DiT':
                kappa_temp = self.diffusionCoef(t, x) 
            else:
                kappa_temp = self.diffusionCoef(x, t) 
            # Global average pooling to get one value per channel
            kappa_temp = F.adaptive_avg_pool2d(kappa_temp, (1, 1))  # Shape: [batch_size, channels, 1, 1]
            # Apply activation to these channel-wise coefficients
            kappa = self.activation(kappa_temp.view(batch_size, channels))  # Shape: [batch_size, channels]
            # Expand to match spatial dimensions for multiplication
            kappa_expanded = kappa.view(batch_size, channels, 1, 1).expand(-1, -1, h, w)
            # Apply diffusion
            # Alternative version
            kappa_expanded_flat = kappa_expanded.reshape(batch_size, -1)
            diffusion_term = kappa_expanded_flat * torch.mm(L, x_flat)

            reaction_term = z0.view(batch_size, -1)
            z1_flat = diffusion_term + reaction_term # plot contributions curve

            z1 = z1_flat.view(batch_size, channels, h, w)

            return z1, diffusion_term.abs().max(), reaction_term.abs().max()
            
        else:

            return z0, 1e-6+torch.zeros(1,1), z0.abs().max()
              

class ReactionDiffusion7_forNonLinearDiffusion(nn.Module):
    """
        # Setup the system
        # dX/dt = L(k,t)@X + f(X,t)      # if f(X,t) = grad phi(X,t)  <=> min 0.5*x^L*x + phi(x,t)
        # dx1/dt = k1*L*x1 + f1(x1,x2,t)
        # dx2/dt = k2*L*x2 + f2(x1,x2,t)
    """
    def __init__(self, 
                base_flow_network,
                in_channels=4, 
                adj_mode='attention',
                learnable_adj=True,
                diffusion=True,
                identity_Laplacian=False,
                activation_type='ELU',  # New parameter for activation function
                diffusion_network = resnet,
                time_channels = 32,
                time_hidden_channels=32,
                time_num_frequencies=8 
                ):
        super(ReactionDiffusion7_forNonLinearDiffusion, self).__init__()
        
        # this network calls a laplacian function that only works with the attention mode of neighbour finding
        # Base flow network (DiT, ResNet, etc.) - produces velocity field Z0
        self.reaction = base_flow_network
        
        self.diffusion = diffusion # whether to use diffusion term or not
        self.identity_Laplacian = identity_Laplacian

        if not self.identity_Laplacian: #edit , was if self.identity_Laplacian
            if self.diffusion:
                if adj_mode == 'attention':
                    self.adj_generator = AdjGenerator_Attention_ASPP_Time(in_channels=in_channels, out_channels=in_channels, mode=adj_mode,
                                                                         n_time_channels=time_channels)  #AdjGenerator_Attention(mode=adj_mode, learnable=learnable_adj)
                    
                else:
                    self.adj_generator = AdjacencyGenerator(mode=adj_mode, learnable=learnable_adj)
        
        if self.diffusion==True:
            self.time_embed = TimeEmbedding(n_channels=time_channels, hidden_dim=time_hidden_channels, num_freqs=time_num_frequencies)
            self.diffusion_term = diffusion_network #resnet(in_channels, in_channels, diffusion_resnet_channels, num_layers=3) #128, num_layers=3


        if activation_type == 'Softmax':
            self.activation = nn.Softmax(dim=1)  # Now over channels instead of spatial dimensions
        elif activation_type == 'SiLU':
            self.activation = nn.SiLU()
        elif activation_type == 'ReLU':
            self.activation = nn.ReLU()
        else:  # Default is ELU
            self.activation = nn.ELU()
        
    def forward(self, x, t):
        # change the above to (self, x, t) if not using DiT as the base model
        # Z0 = DiT(X) - base flow computation that produces velocity field
        # Shape: [batch_size, channels, h, w]
        # z0 = self.reaction(x, t) # if not using DiT
        # z0 = self.reaction(t, x) # if using DiT


        reaction_term = self.reaction(x, t) # if not using DiT

        ####################################################################
        if self.diffusion==True:

            # Modified implementation (one coefficient per channel per batch):
            diffusion_term = self.diffusion_term(x, t)  # this should be the NonLinearHeatFlow network, which only outputs one tensor

            output = diffusion_term + reaction_term # plot contributions curve

            return output, diffusion_term.abs().max(), reaction_term.abs().max()
            
        else:

            return reaction_term, 1e-6+torch.zeros(1,1), 1e-6+torch.zeros(1,1)
        

class ReactionDiffusion7_forNonLinearDiffusion2(nn.Module):
    """
        # Setup the system
        # dX/dt = L(k,t)@X + f(X,t)      # if f(X,t) = grad phi(X,t)  <=> min 0.5*x^L*x + phi(x,t)
        # dx1/dt = k1*L*x1 + f1(x1,x2,t)
        # dx2/dt = k2*L*x2 + f2(x1,x2,t)
    """
    def __init__(self, 
                base_flow_network,
                diffusion=True,
                activation_type='ELU',  # New parameter for activation function
                diffusion_network = resnet
                ):
        super(ReactionDiffusion7_forNonLinearDiffusion2, self).__init__()
        
        # this network calls a laplacian function that only works with the attention mode of neighbour finding
        # Base flow network (DiT, ResNet, etc.) - produces velocity field Z0
        self.reaction = base_flow_network
        
        self.diffusion = diffusion # whether to use diffusion term or not

        # if not self.identity_Laplacian: #edit , was if self.identity_Laplacian
        #     if self.diffusion:
        #         if adj_mode == 'attention':
        #             self.adj_generator = AdjGenerator_Attention_ASPP_Time(in_channels=in_channels, out_channels=in_channels, mode=adj_mode,
        #                                                                  n_time_channels=time_channels)  #AdjGenerator_Attention(mode=adj_mode, learnable=learnable_adj)
                    
        #         else:
        #             self.adj_generator = AdjacencyGenerator(mode=adj_mode, learnable=learnable_adj)
        
        if self.diffusion==True:
            # self.time_embed = TimeEmbedding(n_channels=time_channels, hidden_dim=time_hidden_channels, num_freqs=time_num_frequencies)
            self.diffusion_term = diffusion_network #resnet(in_channels, in_channels, diffusion_resnet_channels, num_layers=3) #128, num_layers=3


        if activation_type == 'Softmax':
            self.activation = nn.Softmax(dim=1)  # Now over channels instead of spatial dimensions
        elif activation_type == 'SiLU':
            self.activation = nn.SiLU()
        elif activation_type == 'ReLU':
            self.activation = nn.ReLU()
        else:  # Default is ELU
            self.activation = nn.ELU()
        
    def forward(self, x, t):
        # change the above to (self, x, t) if not using DiT as the base model
        # Z0 = DiT(X) - base flow computation that produces velocity field
        # Shape: [batch_size, channels, h, w]
        # z0 = self.reaction(x, t) # if not using DiT
        # z0 = self.reaction(t, x) # if using DiT


        if self.reaction.__class__.__name__ == 'DiT' or self.reaction.__class__.__name__ == 'DhariwalUNet':
            reaction_term = self.reaction(t, x) # if using DiT as the reaction term
        else:
            reaction_term = self.reaction(x, t) # if NOT using DiT as the reaction term


        ####################################################################
        if self.diffusion==True:

            # Modified implementation (one coefficient per channel per batch):
            diffusion_term = self.diffusion_term(x, t)  # this should be the NonLinearHeatFlow network, which only outputs one tensor

            output = diffusion_term + reaction_term # plot contributions curve

            return output, diffusion_term.abs().max(), reaction_term.abs().max()
            
        else:

            return reaction_term, 1e-6+torch.zeros(1,1), reaction_term.abs().max()

# class ReactionDiffusion7_forNonLinearDiffusion2_conditional(nn.Module):
#     """
#     """
#     def __init__(self, 
#                 base_flow_network,
#                 diffusion=True,
#                 activation_type='ELU',  # New parameter for activation function
#                 diffusion_network = resnet
#                 ):
#         super(ReactionDiffusion7_forNonLinearDiffusion2_conditional, self).__init__()
        
#         # this network calls a laplacian function that only works with the attention mode of neighbour finding
#         # Base flow network (DiT, ResNet, etc.) - produces velocity field Z0
#         self.reaction = base_flow_network
        
#         self.diffusion = diffusion # whether to use diffusion term or not

#         # if not self.identity_Laplacian: #edit , was if self.identity_Laplacian
#         #     if self.diffusion:
#         #         if adj_mode == 'attention':
#         #             self.adj_generator = AdjGenerator_Attention_ASPP_Time(in_channels=in_channels, out_channels=in_channels, mode=adj_mode,
#         #                                                                  n_time_channels=time_channels)  #AdjGenerator_Attention(mode=adj_mode, learnable=learnable_adj)
                    
#         #         else:
#         #             self.adj_generator = AdjacencyGenerator(mode=adj_mode, learnable=learnable_adj)
        
#         if self.diffusion==True:
#             # self.time_embed = TimeEmbedding(n_channels=time_channels, hidden_dim=time_hidden_channels, num_freqs=time_num_frequencies)
#             self.diffusion_term = diffusion_network #resnet(in_channels, in_channels, diffusion_resnet_channels, num_layers=3) #128, num_layers=3


#         if activation_type == 'Softmax':
#             self.activation = nn.Softmax(dim=1)  # Now over channels instead of spatial dimensions
#         elif activation_type == 'SiLU':
#             self.activation = nn.SiLU()
#         elif activation_type == 'ReLU':
#             self.activation = nn.ReLU()
#         else:  # Default is ELU
#             self.activation = nn.ELU()
        
#     def forward(self, x, t, y=None, cfg_scale=1.0):
#         # y: labels 
#         # change the above to (self, x, t) if not using DiT as the base model
#         # Z0 = DiT(X) - base flow computation that produces velocity field
#         # Shape: [batch_size, channels, h, w]
#         # z0 = self.reaction(x, t) # if not using DiT
#         # z0 = self.reaction(t, x) # if using DiT


#         if self.reaction.__class__.__name__ == 'DiT' or self.reaction.__class__.__name__ == 'DhariwalUNet':
#             reaction_term = self.reaction.forward_with_cfg(noise_label=t, x=x, y=y, cfg_scale=cfg_scale) # if using DiT as the reaction term
#         else:
#             reaction_term = self.reaction(x, t) # if NOT using DiT as the reaction term


#         ####################################################################
#         if self.diffusion==True:

#             # Modified implementation (one coefficient per channel per batch):
#             diffusion_term = self.diffusion_term(x, t)  # this should be the NonLinearHeatFlow network, which only outputs one tensor

#             output = diffusion_term + reaction_term # plot contributions curve

#             return output, diffusion_term.abs().max(), reaction_term.abs().max()
            
#         else:

#             return reaction_term, 1e-6+torch.zeros(1,1), reaction_term.abs().max()
        
# class ReactionDiffusion7_forNonLinearDiffusion2_conditional(nn.Module):
#     def __init__(self, base_flow_network, diffusion=True, activation_type='ELU', diffusion_network=None):
#         super().__init__()
#         self.reaction = base_flow_network
#         self.diffusion = diffusion
#         self.diffusion_term = diffusion_network if diffusion else None
#         self.activation = nn.ELU() if activation_type not in ['Softmax','SiLU','ReLU'] else \
#             {'Softmax': nn.Softmax(dim=1), 'SiLU': nn.SiLU(), 'ReLU': nn.ReLU()}[activation_type]

#     def forward(self, x, t, y=None, cfg_scale=1.0):
#         # --- Reaction ---
#         if hasattr(self.reaction, "forward_with_cfg") and not self.training and cfg_scale > 1.0 and y is not None:
#             v_reac = self.reaction.forward_with_cfg(noise_labels=t, x=x, y=y, cfg_scale=cfg_scale)
#         else:
#             # DhariwalUNet/DiT: training path or no-CFG inference
#             if self.reaction.__class__.__name__ in ['DiT','DhariwalUNet']:
#                 v_reac = self.reaction(t, x=x, y=y)
#             else:
#                 v_reac = self.reaction(x, t)

#         # --- Diffusion (GPS) ---
#         if self.diffusion and self.diffusion_term is not None:
#             if hasattr(self.diffusion_term, "forward_with_cfg") and not self.training and cfg_scale > 1.0 and y is not None:
#                 v_diff = self.diffusion_term.forward_with_cfg(x, t, y, cfg_scale=cfg_scale)
#             else:
#                 v_diff = self.diffusion_term(x, t, y=y)
#             out = v_reac + v_diff
#             return out, v_diff.abs().max(), v_reac.abs().max()
#         else:
#             return v_reac, torch.tensor(1e-6, device=x.device), v_reac.abs().max()


class ReactionDiffusion7_forNonLinearDiffusion2_conditional(nn.Module):
    def __init__(self, base_flow_network, diffusion=True, activation_type='ELU', diffusion_network=None):
        super().__init__()
        self.reaction = base_flow_network
        self.diffusion = diffusion
        self.diffusion_term = diffusion_network if diffusion else None
        self.activation = (
            {'Softmax': nn.Softmax(dim=1), 'SiLU': nn.SiLU(), 'ReLU': nn.ReLU()}.get(activation_type, nn.ELU())
        )

    @staticmethod
    def _is_dit(model) -> bool:
        # Works with your provided DiT class names (e.g., "DiT", "DiT_B_2", etc.)
        name = model.__class__.__name__
        return (name == "DiT") or ("DiT" in name)

    @staticmethod
    def _is_dhariwal_unet(model) -> bool:
        return model.__class__.__name__ == "DhariwalUNet"

    def _reaction_forward(self, x, t, y, cfg_scale):
        """
        Handles both CFG (in eval + cfg_scale>1 + y not None) and plain forward, 
        with the correct calling conventions for DhariwalUNet and DiT.
        """
        use_cfg = (not self.training) and (cfg_scale is not None) and (cfg_scale > 1.0) and (y is not None) and hasattr(self.reaction, "forward_with_cfg")

        if self._is_dit(self.reaction):
            if use_cfg:
                # DiT: forward_with_cfg(t, x, y, cfg_scale=...)
                return self.reaction.forward_with_cfg(t, x, y, cfg_scale=cfg_scale)
            else:
                # DiT: forward(t, x, y)
                return self.reaction(t, x, y)
        elif self._is_dhariwal_unet(self.reaction):
            if use_cfg:
                # DhariwalUNet: forward_with_cfg(noise_labels=t, x=x, y=y, cfg_scale=...)
                return self.reaction.forward_with_cfg(noise_labels=t, x=x, y=y, cfg_scale=cfg_scale)
            else:
                # DhariwalUNet: forward(noise_labels=t, x=x, y=y)
                return self.reaction(noise_labels=t, x=x, y=y)
        else:
            # Generic fallback: common U-Net style (x, t) or (x, t, y) if it supports y
            try:
                return self.reaction(x, t, y=y)
            except TypeError:
                return self.reaction(x, t)

    def _diffusion_forward(self, x, t, y, cfg_scale):
        """
        GPS conditional diffusion block (or any other diffusion module).
        If it exposes forward_with_cfg, use it during eval with cfg_scale>1 and labels available.
        """
        if self.diffusion and (self.diffusion_term is not None):
            use_cfg = (not self.training) and (cfg_scale is not None) and (cfg_scale > 1.0) and (y is not None) and hasattr(self.diffusion_term, "forward_with_cfg")
            if use_cfg:
                # GPS conditional block: forward_with_cfg(x, t, y, cfg_scale=...)
                return self.diffusion_term.forward_with_cfg(x, t, y, cfg_scale=cfg_scale)
            else:
                # GPS conditional block: forward(x, t, y=...)
                try:
                    return self.diffusion_term(x, t, y=y)
                except TypeError:
                    # Fall back if diffusion term ignores labels
                    return self.diffusion_term(x, t)
        return None

    def forward(self, x, t, y=None, cfg_scale=1.0):
        # --- Reaction term ---
        v_reac = self._reaction_forward(x, t, y, cfg_scale)

        # --- Diffusion (GPS) term ---
        v_diff = self._diffusion_forward(x, t, y, cfg_scale)

        if v_diff is not None:
            out = v_reac + v_diff
            return out, v_diff.abs().max(), v_reac.abs().max()
        else:
            return v_reac, torch.tensor(1e-6, device=x.device), v_reac.abs().max()


class ReactionDiffusion_AttnFlow(nn.Module):
    """
        # Setup the system
        # dX/dt = L(k,t)@X + f(X,t)      # if f(X,t) = grad phi(X,t)  <=> min 0.5*x^L*x + phi(x,t)
        # dx1/dt = k1*L*x1 + f1(x1,x2,t)
        # dx2/dt = k2*L*x2 + f2(x1,x2,t)
    """
    def __init__(self, 
                in_channels=4, 
                adj_mode='attention',
                learnable_adj=True,
                diffusion=True,
                activation_type='ELU',  # New parameter for activation function
                diffusion_network = resnet,
                time_channels = 32,
                time_hidden_channels=32,
                time_num_frequencies=8 
                ):
        super(ReactionDiffusion_AttnFlow, self).__init__()
        
        # this network calls a laplacian function that only works with the attention mode of neighbour finding
        # Base flow network (DiT, ResNet, etc.) - produces velocity field Z0

        self.diffusion = diffusion # whether to use diffusion term or not
        self.adj_mode = adj_mode

        if self.diffusion:
            if adj_mode == 'attention':
                self.time_embed = TimeEmbedding(n_channels=time_channels, hidden_dim=time_hidden_channels, num_freqs=time_num_frequencies)
                self.adj_generator = AdjGenerator_Attention_ASPP_Time(in_channels=in_channels, out_channels=in_channels, mode=adj_mode,
                                                                         n_time_channels=time_channels)            
            else:
                self.adj_generator = AdjacencyGenerator(mode=adj_mode, learnable=learnable_adj)

            self.diffusionCoef = diffusion_network #resnet(in_channels, in_channels, diffusion_resnet_channels, num_layers=3) #128, num_layers=3

        if activation_type == 'Softmax':
            self.activation = nn.Softmax(dim=1)  # Now over channels instead of spatial dimensions
        elif activation_type == 'SiLU':
            self.activation = nn.SiLU()
        elif activation_type == 'ReLU':
            self.activation = nn.ReLU()
        else:  # Default is ELU
            self.activation = nn.ELU()
        
    def forward(self, x, t):
        # change the above to (self, x, t) if not using DiT as the base model
        # Z0 = DiT(X) - base flow computation that produces velocity field
        # Shape: [batch_size, channels, h, w]
        # z0 = self.reaction(x, t) # if not using DiT
        # z0 = self.reaction(t, x) # if using DiT

        ####################################################################
        if self.diffusion==True:
            if self.adj_mode=='attention':
                t_embedded = self.time_embed(t)
                # Compute graph Laplacian based on current adjacency mode
                L = compute_graph_laplacian_with_timeEmbed(x, t_embedded, self.adj_generator)
            else:
                L = compute_graph_laplacian(x, self.adj_generator)
            
            # # Reshape for graph operation
            batch_size, channels, h, w = x.shape
            x_flat = x.view(batch_size, -1)  # [batch_size, channels*h*w]
            
            # kappa = self.activation(self.diffusionCoef(x, t))
            # kappa = self.diffusionCoef(x, t)  # try: no activation

            # Modified implementation (one coefficient per channel per batch):
            if self.diffusionCoef.__class__.__name__ == 'DiT':
                kappa_temp = self.diffusionCoef(t, x) 
            else:
                kappa_temp = self.diffusionCoef(x, t) 
            # Global average pooling to get one value per channel
            kappa_temp = F.adaptive_avg_pool2d(kappa_temp, (1, 1))  # Shape: [batch_size, channels, 1, 1]
            # Apply activation to these channel-wise coefficients
            kappa = self.activation(kappa_temp.view(batch_size, channels))  # Shape: [batch_size, channels]
            # Expand to match spatial dimensions for multiplication
            kappa_expanded = kappa.view(batch_size, channels, 1, 1).expand(-1, -1, h, w)
            # Apply diffusion
            # Alternative version
            kappa_expanded_flat = kappa_expanded.reshape(batch_size, -1)
            diffusion_term = kappa_expanded_flat * torch.mm(L, x_flat)

            z1_flat = diffusion_term 

            z1 = z1_flat.view(batch_size, channels, h, w)

            return z1
            



def compute_weighted_gradient(N1_X, A):
    """
    Efficiently compute weighted gradient without explicitly forming G.
    
    Args:
        N1_X: Tensor, shape [B, F], output of N1
        A: Adjacency matrix, shape [B, B], differentiable
    
    Returns:
        GX: Weighted gradients, shape [B, B, F]
    """
    # Compute differences between all pairs (efficient broadcasting)
    GX = N1_X.unsqueeze(1) - N1_X.unsqueeze(0)  # [B, B, F]
    
    # Apply adjacency weighting (sqrt if desired)
    weighted_GX = torch.sqrt(A.unsqueeze(-1)) * GX  # [B, B, F]

    return weighted_GX  # [B, B, F]

def compute_weighted_divergence(activated_GX, A):
    """
    Efficiently compute divergence (G^T @ activated_GX) without sparse ops.
    
    Args:
        activated_GX: Tensor after activation, shape [B, B, F]
        A: Adjacency matrix, shape [B, B]
    
    Returns:
        divergence: Tensor, shape [B, F]
    """
    weighted_activated = torch.sqrt(A.unsqueeze(-1)) * activated_GX  # [B, B, F]
    
    divergence = weighted_activated.sum(dim=1) - weighted_activated.sum(dim=0)  # [B, F]
    
    return divergence  # [B, F]


class NonLinearHeatFlow(nn.Module):
    ''' 
    % Let's Define the graph gradient operator G acting on z = N_1(x,t):
    \[
    \bigl( G\,z \bigr)_{ij} = \sqrt{A_{ij}}\,(z_i - z_j)
    \]
    and its adjoint \(G^T\):
    \[
    \bigl( G^T Q \bigr)_i = \sum_{j=1}^{B} \sqrt{A_{ij}}\, Q_{ij} - \sum_{j=1}^{B} \sqrt{A_{ji}}\, Q_{ji}\,.
    \]
    Then, the PDE implemented by the network is:
    \[
    \frac{\partial x}{\partial t} = -\,N_2\Bigl( G^T\Bigl[\sigma\bigl(G\,N_1(x,t)\bigr)\Bigr],\, t\Bigr)\,.
    \]
    
    '''
    def __init__(self, inner_model, outer_model, in_channels=4, adj_mode='attention', activation_type='ELU', time_channels=64, time_hidden_channels=64, time_num_frequencies=64):
        super().__init__()
        self.N1 = inner_model
        self.N2 = outer_model
        self.adj_mode = adj_mode


        if adj_mode == 'attention':
                self.time_embed = TimeEmbedding(n_channels=time_channels, hidden_dim=time_hidden_channels, num_freqs=time_num_frequencies)
                self.adj_generator = AdjGenerator_Attention_ASPP_Time(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    mode=adj_mode,
                    n_time_channels=time_channels)
        else:
            self.adj_generator = AdjacencyGenerator(mode=adj_mode, learnable=True)


        if activation_type == 'Softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation_type == 'SiLU':
            self.activation = nn.SiLU()
        elif activation_type == 'ReLU':
            self.activation = nn.ReLU()
        elif activation_type == 'ELU':
            self.activation = nn.ELU()
        else:
            self.activation = nn.SiLU()

        
    def forward(self, x, t):
        B, C, H, W = x.shape
        
        # Step 1: Compute N1(X, theta1)


        if self.N1.__class__.__name__ == 'DiT' or self.N1.__class__.__name__ == 'DhariwalUNet':
            N1_x = self.N1(t, x)  # [B,4,32,32]
        else:
            N1_x = self.N1(x, t)  # [B,4,32,32] # if not using DiT

        N1_x_flat = N1_x.view(B, -1)  # [B, 4096], this is N1(X, theta1)

        # Step 2: Compute adjacency matrix A(X, t) dynamically
        if self.adj_mode=='attention':
            # Embed time for adjacency generation
            t_embedded = self.time_embed(t) if self.time_embed else None
            A = self.adj_generator(x, t_embedded)  # [B, B]
        else:
            A = self.adj_generator(x)

        # Step 3: Efficiently compute weighted gradients G @ N1(X, theta1)
        # Using pairwise differences multiplied by sqrt(A)
        grad_weighted = compute_weighted_gradient(N1_x_flat, A)  # [B,B,4096]

        # Step 4: Apply activation sigma to gradients: sigma(G @ N1(X, theta1))
        activated_grad = self.activation(grad_weighted)  # [B,B,4096]

        # Step 5: Efficiently compute divergence: G^T @ sigma(G @ N1(X, theta1))
        divergence = compute_weighted_divergence(activated_grad, A)  # [B,4096]

        # Step 6: Pass the divergence through outer network N2 (reaction term)
        divergence = divergence.view(B, C, H, W)

        if self.N2.__class__.__name__ == 'DiT' or self.N2.__class__.__name__ == 'DhariwalUNet':
            output = -self.N2(t, divergence)  # [B,4,32,32]  # [B,4,32,32]
        else:
            output = -self.N2(divergence, t)  # [B,4,32,32] # if not using DiT
        

        # Returns output as diffusion velocity, along with max values for debugging
        return output  #, output.abs().max().detach(), divergence.abs().max().detach() # v, diffusion term, reaction term


class NonLinearHeatFlow_IdentityGradientMatrix(nn.Module):
    '''
    Identity-gradient ablation: G = I so that G z = z and
    divergence = sigma(N1(x,t)). Implements:
    \[ \frac{∂x}{∂t} = -N_2(σ(N_1(x,t)),\,t) \]
    '''
    def __init__(self, inner_model, outer_model, activation_type='ELU'):
        super().__init__()
        self.N1 = inner_model
        self.N2 = outer_model
        # No adjacency or laplacian needed for identity gradient

        # Activation σ
        if activation_type == 'Softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation_type == 'SiLU':
            self.activation = nn.SiLU()
        elif activation_type == 'ReLU':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ELU()

    def forward(self, x, t):
        B, C, H, W = x.shape
        # Compute inner network features
        if self.N1.__class__.__name__ in ['DiT', 'DhariwalUNet']:
            N1_x = self.N1(t, x)
        else:
            N1_x = self.N1(x, t)
        # Flatten to [B, C*H*W]
        N1_x_flat = N1_x.view(B, -1)
        # Identity: G z = z => apply σ directly
        activated = self.activation(N1_x_flat)
        # Reshape back to [B, C, H, W]
        divergence = activated.view(B, C, H, W)
        # Outer network N2
        if self.N2.__class__.__name__ in ['DiT', 'DhariwalUNet']:
            out = -self.N2(t, divergence)
        else:
            out = -self.N2(divergence, t)
        return out

    



class ReactionDiffusion7_PixelSpace(nn.Module):
    """
        # Setup the system
        # dX/dt = L(k,t)@X + f(X,t)      # if f(X,t) = grad phi(X,t)  <=> min 0.5*x^L*x + phi(x,t)
        # dx1/dt = k1*L*x1 + f1(x1,x2,t)
        # dx2/dt = k2*L*x2 + f2(x1,x2,t)
    """
    def __init__(self, 
                base_flow_network,
                latent_in_channels=4, 
                adj_mode='attention',
                learnable_adj=True,
                diffusion=True,
                identity_Laplacian=False,
                activation_type='ELU',  # New parameter for activation function
                diffusion_network = resnet,
                time_channels = 32,
                time_hidden_channels=32,
                time_num_frequencies=8 
                ):
        super( ReactionDiffusion7_PixelSpace, self).__init__()
        
        # this network calls a laplacian function that only works with the attention mode of neighbour finding with Time Embedding
        # Base flow network (DiT, ResNet, etc.) - produces velocity field Z0
        self.reaction = base_flow_network
        
        self.diffusion = diffusion # whether to use diffusion term or not
        self.identity_Laplacian = identity_Laplacian

        if not self.identity_Laplacian: #edit , was if self.identity_Laplacian
            if self.diffusion:
                if adj_mode == 'attention':
                    self.adj_generator = AdjGenerator_Attention_ASPP_Time(in_channels=latent_in_channels, out_channels=latent_in_channels, mode=adj_mode,
                                                                         n_time_channels=time_channels)  #AdjGenerator_Attention(mode=adj_mode, learnable=learnable_adj)
                    
                else:
                    self.adj_generator = AdjacencyGenerator(mode=adj_mode, learnable=learnable_adj)
        
        if self.diffusion==True:
            self.time_embed = TimeEmbedding(n_channels=time_channels, hidden_dim=time_hidden_channels, num_freqs=time_num_frequencies)
            self.diffusionCoef = diffusion_network #resnet(in_channels, in_channels, diffusion_resnet_channels, num_layers=3) #128, num_layers=3


        if activation_type == 'Softmax':
            self.activation = nn.Softmax(dim=1)  # Now over channels instead of spatial dimensions
        elif activation_type == 'SiLU':
            self.activation = nn.SiLU()
        elif activation_type == 'ReLU':
            self.activation = nn.ReLU()
        else:  # Default is ELU
            self.activation = nn.ELU()
        
    def forward(self, x, t, enc_x):
        # change the above to (self, x, t) if not using DiT as the base model
        # Z0 = DiT(X) - base flow computation that produces velocity field
        # Shape: [batch_size, channels, h, w]
        z0 = self.reaction(x, t) # if not using DiT
        # z0 = self.reaction(t, x) # if using DiT

        ####################################################################
        if self.diffusion==True:
            if self.identity_Laplacian:
                ######################### TEST with no Laplacian #############
                L = torch.eye(x.shape[0], x.shape[0], device=x.device)
                ######################################################
            else:
                t_embedded = self.time_embed(t)
                # Compute graph Laplacian based on current adjacency mode
                L = compute_graph_laplacian_with_timeEmbed(enc_x, t_embedded, self.adj_generator)
                # L = compute_graph_laplacian(x, self.adj_generator)
            
            # # Reshape for graph operation
            batch_size, channels, h, w = x.shape
            x_flat = x.view(batch_size, -1)  # [batch_size, channels*h*w]
            
            # kappa = self.activation(self.diffusionCoef(x, t))
            # kappa = self.diffusionCoef(x, t)  # try: no activation

            # Modified implementation (one coefficient per channel per batch):
            kappa_temp = self.diffusionCoef(x, t)
            # Global average pooling to get one value per channel
            kappa_temp = F.adaptive_avg_pool2d(kappa_temp, (1, 1))  # Shape: [batch_size, channels, 1, 1]
            # Apply activation to these channel-wise coefficients
            kappa = self.activation(kappa_temp.view(batch_size, channels))  # Shape: [batch_size, channels]
            # Expand to match spatial dimensions for multiplication
            kappa_expanded = kappa.view(batch_size, channels, 1, 1).expand(-1, -1, h, w)
            # Apply diffusion
            # Alternative version
            kappa_expanded_flat = kappa_expanded.reshape(batch_size, -1)
            diffusion_term = kappa_expanded_flat * torch.mm(L, x_flat)

            reaction_term = z0.view(batch_size, -1)
            z1_flat = diffusion_term + reaction_term # plot contributions curve

            z1 = z1_flat.view(batch_size, channels, h, w)

            return z1, diffusion_term.abs().max(), reaction_term.abs().max()
            
        else:

            return z0, 1e-6+torch.zeros(1,1), z0.abs().max()
        

class ReactionDiffusion6_PixelSpace(nn.Module):
    """
        # Setup the system
        # dX/dt = L(k,t)@X + f(X,t)      # if f(X,t) = grad phi(X,t)  <=> min 0.5*x^L*x + phi(x,t)
        # dx1/dt = k1*L*x1 + f1(x1,x2,t)
        # dx2/dt = k2*L*x2 + f2(x1,x2,t)
    """
    def __init__(self, 
                base_flow_network,
                latent_in_channels, 
                adj_mode='attention',
                learnable_adj=True,
                diffusion=True,
                identity_Laplacian=False,
                activation_type='ELU',  # New parameter for activation function
                diffusion_network = resnet 
                ):
        super(ReactionDiffusion6_PixelSpace, self).__init__()
        # NOTE: This network can apply attention but without time embedding 
        # Base flow network (DiT, ResNet, etc.) - produces velocity field Z0
        self.reaction = base_flow_network
        
        self.diffusion = diffusion # whether to use diffusion term or not
        self.identity_Laplacian = identity_Laplacian

        if not self.identity_Laplacian: #edit , was if self.identity_Laplacian
            if self.diffusion==True:
                if adj_mode == 'attention':
                    # self.adj_generator = AdjGenerator_Attention(mode=adj_mode, learnable=learnable_adj)
                    # computes attention WITHOUT time embedding
                    self.adj_generator = AdjGenerator_Attention_ASPP_Time(in_channels=latent_in_channels, out_channels=latent_in_channels, mode=adj_mode,  n_time_channels=None) 
                else:
                    self.adj_generator = AdjacencyGenerator(mode=adj_mode, learnable=learnable_adj)
        
        if self.diffusion==True:
            self.diffusionCoef = diffusion_network #resnet(in_channels, in_channels, diffusion_resnet_channels, num_layers=3) #128, num_layers=3


        if activation_type == 'Softmax':
            self.activation = nn.Softmax(dim=1)  # Now over channels instead of spatial dimensions
        elif activation_type == 'SiLU':
            self.activation = nn.SiLU()
        elif activation_type == 'ReLU':
            self.activation = nn.ReLU()
        else:  # Default is ELU
            self.activation = nn.ELU()
        
        # self.closing_layer = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x, t, enc_x):
        # change the above to (self, x, t) if not using DiT as the base model
        # Z0 = DiT(X) - base flow computation that produces velocity field
        # Shape: [batch_size, channels, h, w]
        z0 = self.reaction(x, t) # if not using DiT
        # z0 = self.reaction(t, x) # if using DiT

        ####################################################################
        if self.diffusion==True:
            if self.identity_Laplacian:
                ######################### TEST with no Laplacian #############
                L = torch.eye(x.shape[0], x.shape[0], device=x.device)
                ######################################################
            else:
                # Compute graph Laplacian based on current adjacency mode
                L = compute_graph_laplacian(enc_x, self.adj_generator)
            
            # # Reshape for graph operation
            batch_size, channels, h, w = x.shape
            x_flat = x.view(batch_size, -1)  # [batch_size, channels*h*w]
            
            # kappa = self.activation(self.diffusionCoef(x, t))
            # kappa = self.diffusionCoef(x, t)  # try: no activation

            # Modified implementation (one coefficient per channel per batch):
            kappa_temp = self.diffusionCoef(x, t)
            # Global average pooling to get one value per channel
            kappa_temp = F.adaptive_avg_pool2d(kappa_temp, (1, 1))  # Shape: [batch_size, channels, 1, 1]
            # Apply activation to these channel-wise coefficients
            kappa = self.activation(kappa_temp.view(batch_size, channels))  # Shape: [batch_size, channels]
            # Expand to match spatial dimensions for multiplication
            kappa_expanded = kappa.view(batch_size, channels, 1, 1).expand(-1, -1, h, w)
            # Apply diffusion
            kappa_expanded_flat = kappa_expanded.reshape(batch_size, -1)
            diffusion_term = kappa_expanded_flat * torch.mm(L, x_flat)

            reaction_term = z0.view(batch_size, -1)
            z1_flat = diffusion_term + reaction_term # plot contributions curve

            z1 = z1_flat.view(batch_size, channels, h, w)

            return z1, diffusion_term.abs().max(), reaction_term.abs().max()
            
        else:

            return z0, 1e-6+torch.zeros(1,1), z0.abs().max()
        

def create_hybrid_model(base_model, in_channels, hidden_dim=64, model_type="hybrid_gnn", adj_mode='gaussian', 
                        diffusion=True, identity_Laplacian=False, activation_type='ELU', diffusion_network=resnet, 
                        time_channels=32, time_hidden_channels=32, time_num_frequencies=8, knn_k=20, gradient_matrix_is_identity=False):

    if model_type == 'diffusion_reaction3':
        return ReactionDiffusion3(
            base_flow_network=base_model,
            in_channels=in_channels,
            adj_mode=adj_mode,
            learnable_adj=True,
            diffusion=diffusion,
            identity_Laplacian=identity_Laplacian,
            activation_type=activation_type,
            diffusion_resnet_channels = hidden_dim  
        )
    elif model_type == 'diffusion_reaction_4':
        return ReactionDiffusion4(
            base_flow_network=base_model,
            in_channels=in_channels,
            adj_mode=adj_mode,
            learnable_adj=True,
            diffusion=diffusion,
            identity_Laplacian=identity_Laplacian,
            activation_type=activation_type,
            diffusion_network=diffusion_network
        )
    
    elif model_type == 'diffusion_reaction_5':
        return ReactionDiffusion5(
            base_flow_network=base_model,
            in_channels=in_channels,
            adj_mode=adj_mode,
            learnable_adj=True,
            diffusion=diffusion,
            identity_Laplacian=identity_Laplacian,
            activation_type=activation_type,
            diffusion_network=diffusion_network)
    
    elif model_type == 'diffusion_reaction_6':
        # with attention
        # used for all DDPM experiments with AFHQ Cat 256
        return ReactionDiffusion6(
            base_flow_network=base_model,
            in_channels=in_channels,
            adj_mode=adj_mode,
            learnable_adj=True,
            diffusion=diffusion,
            identity_Laplacian=identity_Laplacian,
            activation_type=activation_type,
            diffusion_network=diffusion_network)
    elif model_type == 'diffusion_reaction_6_improved':
        return ReactionDiffusion6_improved(
            base_flow_network=base_model,
            in_channels=in_channels,
            adj_mode=adj_mode,
            learnable_adj=True,
            diffusion=diffusion,
            identity_Laplacian=identity_Laplacian,
            activation_type=activation_type,
            diffusion_network=diffusion_network,
            knn_k=knn_k
            )
    elif model_type == "diffusion_reaction_7_withTimeAndAttention":
        return ReactionDiffusion7(
                base_flow_network=base_model,
                in_channels=in_channels, 
                adj_mode=adj_mode,
                learnable_adj=True,
                diffusion=diffusion,
                identity_Laplacian=identity_Laplacian,
                activation_type=activation_type,  # New parameter for activation function
                diffusion_network = diffusion_network,
                time_channels = time_channels,
                time_hidden_channels=time_hidden_channels,
                time_num_frequencies=time_num_frequencies 
              )
    elif model_type == 'nonlinear_heat_eq' and not gradient_matrix_is_identity:
        return NonLinearHeatFlow( 
            inner_model=base_model, 
            outer_model=diffusion_network,
            adj_mode = adj_mode,
            in_channels=in_channels, 
            time_channels=time_channels,
            time_hidden_channels=time_hidden_channels,
            time_num_frequencies=time_num_frequencies,
            activation_type=activation_type)
    
    elif model_type == 'nonlinear_heat_eq' and gradient_matrix_is_identity:
        print("Using identity gradient matrix!")
        return NonLinearHeatFlow_IdentityGradientMatrix( 
            inner_model=base_model, 
            outer_model=diffusion_network,
            activation_type=activation_type)
    
    elif model_type == 'reaction_diffusion_7_pixelSpace_withTimeAndAttention':
        return ReactionDiffusion7_PixelSpace(
            base_flow_network=base_model,
            in_channels=in_channels, 
            adj_mode=adj_mode,
            learnable_adj=True,
            diffusion=diffusion,
            identity_Laplacian=identity_Laplacian,
            activation_type=activation_type,  # New parameter for activation function
            diffusion_network = diffusion_network,
            time_channels = time_channels,
            time_hidden_channels=time_hidden_channels,
            time_num_frequencies=time_num_frequencies)
    
    elif model_type == 'reaction_Diffusion_6_pixelSpace':
        return ReactionDiffusion6_PixelSpace(
            base_flow_network=base_model,
            latent_in_channels=in_channels,
            adj_mode=adj_mode,
            learnable_adj=True,
            diffusion=diffusion,
            identity_Laplacian=identity_Laplacian,
            activation_type=activation_type,
            diffusion_network=diffusion_network)
    
    elif model_type == 'nonLinearHeatDiffusion':

        return ReactionDiffusion7_forNonLinearDiffusion(
                base_flow_network=base_model,
                in_channels=in_channels, 
                adj_mode=adj_mode,
                learnable_adj=True,
                diffusion=diffusion,
                identity_Laplacian=identity_Laplacian,
                activation_type=activation_type,  # New parameter for activation function
                diffusion_network = diffusion_network,
                time_channels = time_channels,
                time_hidden_channels=time_hidden_channels,
                time_num_frequencies=time_num_frequencies 
                )
    elif model_type == 'nonLinearHeatDiffusion2':
   
        return ReactionDiffusion7_forNonLinearDiffusion2(
                    base_flow_network=base_model,
                    diffusion=diffusion,
                    activation_type=activation_type,  # New parameter for activation function
                    diffusion_network = diffusion_network
                    )
    elif model_type == 'nonLinearHeatDiffusion2_conditional':

        return ReactionDiffusion7_forNonLinearDiffusion2_conditional(
                    base_flow_network=base_model,
                    diffusion=diffusion,
                    activation_type=activation_type,  # New parameter for activation function
                    diffusion_network = diffusion_network
                    )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    

