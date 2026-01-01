import torch
import torch.nn as nn
import torch_geometric as pyg
import torch_geometric.transforms as T
from torch_geometric.nn import GPSConv, GatedGraphConv
from torch_geometric.nn.attention import PerformerAttention
from torch_geometric.data import Data, Batch
from typing import Optional, Dict, Any, List, Tuple


class TimeEmbedding(nn.Module):
    """
    Positional embedding for diffusion time t.
    """
    def __init__(self, n_channels: int, hidden_dim: int = 32, num_freqs: int = 8):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(num_freqs))
        self.fc1 = nn.Linear(2 * num_freqs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_channels)
        self.act = nn.SiLU()
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        freqs = self.freqs.view(1, -1)
        t_proj = t @ freqs
        sin = torch.sin(2 * torch.pi * t_proj)
        cos = torch.cos(2 * torch.pi * t_proj)
        x = torch.cat([sin, cos], dim=-1)
        x = self.act(self.fc1(x))
        return self.fc2(x)


class AuthenticRandomWalkPE(nn.Module):
    """
    Creates authentic random walk positional encodings using PyG's transform.
    This follows the original GPS paper's methodology more closely.
    """
    def __init__(self, walk_length: int = 20):
        super().__init__()
        self.walk_length = walk_length
        self.transform = T.AddRandomWalkPE(walk_length=walk_length, attr_name='pe')
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Creates random walk positional encodings for the batch nodes.
        
        Args:
            x: Node features [B, D]
            edge_index: Edge connections [2, E]
            
        Returns:
            Random walk PE features [B, walk_length]
        """
        B = x.shape[0]
        
        # Create a PyG Data object with the batch as nodes
        data = Data(x=x, edge_index=edge_index)
        
        # Apply the random walk transform
        transformed_data = self.transform(data)
        
        # Extract the PE features
        pe = transformed_data.pe  # [B, walk_length]
        
        return pe


class RandomWalkPE(nn.Module):
    """
    NOTE: Same as AuthenticRandomWalkPE
    Creates authentic random walk positional encodings using PyG's transform.
    This follows the original GPS paper's methodology more closely.
    """
    def __init__(self, walk_length: int = 20):
        super().__init__()
        self.walk_length = walk_length
        self.transform = T.AddRandomWalkPE(walk_length=walk_length, attr_name='pe')
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Creates random walk positional encodings for the batch nodes.
        
        Args:
            x: Node features [B, D]
            edge_index: Edge connections [2, E]
            
        Returns:
            Random walk PE features [B, walk_length]
        """
        B = x.shape[0]
        
        # Create a PyG Data object with the batch as nodes
        data = Data(x=x, edge_index=edge_index)
        
        # Apply the random walk transform
        transformed_data = self.transform(data)
        
        # Extract the PE features
        pe = transformed_data.pe  # [B, walk_length]
        
        return pe
    

class RedrawProjection:
    """
    Manages redrawing of projection matrices for Performer attention.
    """
    def __init__(self, model: nn.Module, redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        
        self.num_last_redraw += 1


# class GPSDiffusion(nn.Module):
#     """
#     GPSConv-based diffusion term v_diff(x, t) with authentic RWPE and time concatenation.
    
#     Inputs:
#     - x: Tensor [B, C, H, W]
#     - t: Tensor [B] or [B,1]
    
#     Output:
#     - v: Tensor [B, C, H, W]
#     """
#     def __init__(
#         self,
#         in_channels: int = 4,
#         H: int = 32,
#         W: int = 32,
#         hidden_dim: int = 64,
#         pe_dim: int = 16,
#         walk_length: int = 20,
#         num_layers: int = 5,
#         heads: int = 4,
#         attn_type: str = 'multihead',
#         attn_kwargs: dict = None,
#         time_channels: int = 64,
#         time_hidden: int = 64,
#         time_freqs: int = 8,
#         redraw_interval: int = 1000
#     ):
#         super().__init__()
#         D = in_channels * H * W
        
#         # 1) Data projection
#         self.node_proj = nn.Linear(D, hidden_dim)
        
#         # 2) Time embedding + concatenation projection
#         self.time_embed = TimeEmbedding(time_channels, time_hidden, time_freqs)
#         self.time_proj = nn.Linear(time_channels, hidden_dim)
#         self.time_cat_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
#         # 3) Random Walk PE (using PyG's authentic implementation)
#         self.pe_generator = AuthenticRandomWalkPE(walk_length=walk_length)
#         self.pe_norm = nn.BatchNorm1d(walk_length)
#         self.pe_lin = nn.Linear(walk_length, pe_dim)
#         self.cat_proj = nn.Linear(hidden_dim + pe_dim, hidden_dim)
        
#         # 4) GPSConv stack
#         if attn_kwargs is None:
#             attn_kwargs = {'dropout': 0.1}
        
#         self.convs = nn.ModuleList()
#         for _ in range(num_layers):
#             gated = GatedGraphConv(hidden_dim, num_layers=1)
#             conv = GPSConv(
#                 hidden_dim,
#                 gated,
#                 heads=heads,
#                 attn_type=attn_type,
#                 attn_kwargs=attn_kwargs
#             )
#             self.convs.append(conv)
        
#         # 5) Output projection
#         self.out_proj = nn.Linear(hidden_dim, D)
        
#         # 6) RedrawProjection handler for performer attention
#         self.redraw_projection = RedrawProjection(
#             self,
#             redraw_interval=redraw_interval if attn_type == 'performer' else None
#         )
        
#         # Store dimensions for reshaping
#         self.in_channels = in_channels
#         self.H = H
#         self.W = W
        
#     def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
#         # x: [B, C, H, W], t: [B] or [B, 1]
#         B, C, H, W = x.shape
        
#         # Data projection for node features
#         x_flat = x.view(B, -1)  # [B, C*H*W]
#         h_data = self.node_proj(x_flat)  # [B, hidden_dim]
        
#         # Time embedding and concatenation
#         t_emb = self.time_embed(t)  # [B, time_channels]
#         h_time = self.time_proj(t_emb)  # [B, hidden_dim]
#         h = torch.cat([h_data, h_time], dim=-1)  # [B, hidden_dim*2]
#         h = self.time_cat_proj(h)  # [B, hidden_dim]
        
#         # Create fully-connected graph structure
#         # Each batch item is a node, with edges to all other nodes
#         idx = torch.arange(B, device=x.device)
#         src = idx.repeat_interleave(B)
#         dst = idx.repeat(B)
#         edge_index = torch.stack([src, dst], dim=0)  # [2, B*B]
        
#         # Generate authentic random walk positional encodings
#         # We pass the node features and graph structure to the PE generator
#         pe = self.pe_generator(h, edge_index)  # [B, walk_length]
        
#         # Process positional encodings
#         pe_normed = self.pe_norm(pe)  # [B, walk_length]
#         h_pe = self.pe_lin(pe_normed)  # [B, pe_dim]
        
#         # Combine node features with positional encodings
#         h = torch.cat([h, h_pe], dim=-1)  # [B, hidden_dim+pe_dim]
#         h = self.cat_proj(h)  # [B, hidden_dim]
        
#         # Prepare batch indices for PyG
#         batch = torch.arange(B, device=x.device)
        
#         # Redraw projections if using performer attention
#         self.redraw_projection.redraw_projections()
        
#         # GPS message-passing
#         for conv in self.convs:
#             h = conv(h, edge_index, batch)  # [B, hidden_dim]
        
#         # Project back to original dimensions
#         out = self.out_proj(h)  # [B, C*H*W]
#         return out.view(B, C, H, W)  # [B, C, H, W]

class GPSDiffusion2(nn.Module):
    """
    This is the GPS network.
    GPSConv-based diffusion term v_diff(x, t) with authentic RWPE and time concatenation.
    
    Inputs:
    - x: Tensor [B, C, H, W]
    - t: Tensor [B] or [B,1]
    
    Output:
    - v: Tensor [B, C, H, W] a
    """
    def __init__(
        self,
        in_channels: int = 4,
        H: int = 32,
        W: int = 32,
        hidden_dim: int = 64,
        pe_dim: int = 16,
        walk_length: int = 20,
        num_layers: int = 5,
        heads: int = 4,
        attn_type: str = 'multihead',
        attn_kwargs: dict = None,
        time_channels: int = 64,
        time_hidden: int = 64,
        time_freqs: int = 8,
        redraw_interval: int = 1000,
        use_identity_graph: bool = False
    ):
        super().__init__()
        D = in_channels * H * W
        
        self.use_identity_graph = use_identity_graph

        # 1) Data projection
        self.node_proj = nn.Linear(D, hidden_dim)
        
        # 2) Time embedding + concatenation projection
        self.time_embed = TimeEmbedding(time_channels, time_hidden, time_freqs)
        self.time_proj = nn.Linear(time_channels, hidden_dim)
        self.time_cat_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 3) Random Walk PE (using PyG's authentic implementation)
        self.pe_generator = RandomWalkPE(walk_length=walk_length)
        self.pe_norm = nn.BatchNorm1d(walk_length)
        self.pe_lin = nn.Linear(walk_length, pe_dim)
        self.cat_proj = nn.Linear(hidden_dim + pe_dim, hidden_dim)
        
        # 4) GPSConv stack
        if attn_kwargs is None:
            attn_kwargs = {'dropout': 0.1}
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            gated = GatedGraphConv(hidden_dim, num_layers=1)
            conv = GPSConv(
                hidden_dim,
                gated,
                heads=heads,
                attn_type=attn_type,
                attn_kwargs=attn_kwargs
            )
            self.convs.append(conv)
        
        # 5) Output projection
        self.out_proj = nn.Linear(hidden_dim, D)
        
        # 6) RedrawProjection handler for performer attention
        self.redraw_projection = RedrawProjection(
            self,
            redraw_interval=redraw_interval if attn_type == 'performer' else None
        )
        
        # Store dimensions for reshaping
        self.in_channels = in_channels
        self.H = H
        self.W = W
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], t: [B] or [B, 1]
        B, C, H, W = x.shape
        
        # Data projection for node features
        x_flat = x.view(B, -1)  # [B, C*H*W]
        h_data = self.node_proj(x_flat)  # [B, hidden_dim]
        
        # Time embedding and concatenation
        t_emb = self.time_embed(t)  # [B, time_channels]
        h_time = self.time_proj(t_emb)  # [B, hidden_dim]
        h = torch.cat([h_data, h_time], dim=-1)  # [B, hidden_dim*2]
        h = self.time_cat_proj(h)  # [B, hidden_dim]
        
        # Create fully-connected graph structure
        # Each batch item is a node, with edges to all other nodes
        # Create graph structure + batch grouping
        idx = torch.arange(B, device=x.device)
        if self.use_identity_graph:
            # Identity adjacency: self-loops only, and isolate nodes in batch
            edge_index = torch.stack([idx, idx], dim=0)       # [2, B]
            batch = idx                                       # [B], each node its own graph (no inter-node communication)
        else:
            # Here we build a fully-connected graph, and place all nodes in one batch graph so that global communication is possible
            src = idx.repeat_interleave(B)
            dst = idx.repeat(B)
            edge_index = torch.stack([src, dst], dim=0)       # [2, B*B]
            batch = torch.zeros(B, dtype=torch.long, device=x.device)  # one graph

        
        # Generate authentic random walk positional encodings
        # We pass the node features and graph structure to the PE generator
        pe = self.pe_generator(h, edge_index)  # [B, walk_length]
        
        # Process positional encodings
        pe_normed = self.pe_norm(pe)  # [B, walk_length]
        h_pe = self.pe_lin(pe_normed)  # [B, pe_dim]
        
        # Combine node features with positional encodings
        h = torch.cat([h, h_pe], dim=-1)  # [B, hidden_dim+pe_dim]
        h = self.cat_proj(h)  # [B, hidden_dim]
        
        # Redraw projections if using performer attention
        self.redraw_projection.redraw_projections()
        
        # GPS message-passing
        for conv in self.convs:
            h = conv(h, edge_index, batch)  # [B, hidden_dim]
        
        # Project back to original dimensions
        out = self.out_proj(h)  # [B, C*H*W]
        return out.view(B, C, H, W)  # [B, C, H, W]
    
###################################################
# Conditional GPS 
class GPSDiffusion2_Conditional(nn.Module):
    """
    Conditional GPS diffusion v_diff(x,t | y) with:
      - class embedding + optional label_dropout (Alg. 3)
      - forward_with_cfg for classifier-free guidance (Alg. 4)
    """
    def __init__(
        self,
        in_channels: int = 4, H: int = 32, W: int = 32,
        hidden_dim: int = 64, pe_dim: int = 16, walk_length: int = 20,
        num_layers: int = 5, heads: int = 4, attn_type: str = 'multihead',
        attn_kwargs: dict = None,
        time_channels: int = 64, time_hidden: int = 64, time_freqs: int = 8,
        redraw_interval: int = 1000, use_identity_graph: bool = False,
        # --- NEW: conditioning ---
        label_dim: int = 0,             # 0 => unconditional
        label_dropout: float = 0.0,     # e.g., 0.1 like DiT in LFM
    ):
        super().__init__()
        D = in_channels * H * W
        self.in_channels, self.H, self.W = in_channels, H, W
        self.use_identity_graph = use_identity_graph
        self.label_dim = label_dim
        self.label_dropout = float(label_dropout)

        # 1) data projection
        self.node_proj = nn.Linear(D, hidden_dim)

        # 2) time embedding
        self.time_embed = TimeEmbedding(time_channels, time_hidden, time_freqs)
        self.time_proj  = nn.Linear(time_channels, hidden_dim)

        # 3) label embedding (conditional)
        if label_dim and label_dim > 0:
            self.label_embed = nn.Embedding(label_dim + 1, hidden_dim)  # +1 for null idx = label_dim
            self.null_label_idx = label_dim
        else:
            self.label_embed = None
            self.null_label_idx = None

        # fuse (data,time,label) -> hidden
        self.fuse_proj = nn.Linear(hidden_dim * (2 if self.label_embed is None else 3), hidden_dim)

        # 4) RW positional encodings
        self.pe_generator = RandomWalkPE(walk_length=walk_length)
        self.pe_norm = nn.BatchNorm1d(walk_length)
        self.pe_lin  = nn.Linear(walk_length, pe_dim)
        self.cat_proj = nn.Linear(hidden_dim + pe_dim, hidden_dim)

        # 5) GPSConv stack
        if attn_kwargs is None: attn_kwargs = {'dropout': 0.1}
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            gated = GatedGraphConv(hidden_dim, num_layers=1)
            self.convs.append(GPSConv(hidden_dim, gated, heads=heads, attn_type=attn_type, attn_kwargs=attn_kwargs))

        # 6) out proj
        self.out_proj = nn.Linear(hidden_dim, D)

        # 7) performer redraw helper
        self.redraw_projection = RedrawProjection(self, redraw_interval if attn_type == 'performer' else None)

    def _apply_label_dropout(self, y: torch.Tensor, drop_half_label: bool = False) -> torch.Tensor:
        """
        Implements Alg. 3 style nulling:
          - if training and label_dropout>0: randomly set some labels to null
          - if drop_half_label: force second half to null (used when batching cond/uncond)
        """
        if self.label_embed is None or y is None:
            return None

        y_eff = y.clone()
        if self.training and self.label_dropout > 0.0:
            mask = (torch.rand(y_eff.shape[0], device=y_eff.device) < self.label_dropout)
            y_eff[mask] = self.null_label_idx

        if drop_half_label:
            half = y_eff.shape[0] // 2
            y_eff[half:] = self.null_label_idx
        return y_eff

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None,
                drop_half_label: bool = False) -> torch.Tensor:
        B, C, H, W = x.shape
        x_flat = x.view(B, -1)
        h_data = self.node_proj(x_flat)

        t_emb = self.time_embed(t)           # [B, time_channels]
        h_time = self.time_proj(t_emb)       # [B, hidden_dim]

        feats = [h_data, h_time]
        if self.label_embed is not None and y is not None:
            y_eff = self._apply_label_dropout(y, drop_half_label)
            h_lbl = self.label_embed(y_eff)
            feats.append(h_lbl)

        h = torch.cat(feats, dim=-1)
        h = self.fuse_proj(h)                # [B, hidden_dim]

        # graph
        # Create graph structure + batch grouping
        idx = torch.arange(B, device=x.device)
        if self.use_identity_graph:
            # Identity adjacency: self-loops only, and isolate nodes in batch
            edge_index = torch.stack([idx, idx], dim=0)       # [2, B]
            batch = idx                                       # [B], each node its own graph
        else:
            # Fully-connected graph, and place all nodes in one batch graph
            src = idx.repeat_interleave(B)
            dst = idx.repeat(B)
            edge_index = torch.stack([src, dst], dim=0)       # [2, B*B]
            batch = torch.zeros(B, dtype=torch.long, device=x.device)  # one graph


        # RWPE
        pe = self.pe_generator(h, edge_index)      # [B, walk_length]
        pe = self.pe_norm(pe)
        h_pe = self.pe_lin(pe)
        h = self.cat_proj(torch.cat([h, h_pe], dim=-1))

        # GPS stack
        self.redraw_projection.redraw_projections()
        for conv in self.convs:
            h = conv(h, edge_index, batch)

        out = self.out_proj(h).view(B, C, H, W)
        return out

    @torch.no_grad()
    def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor],
                         cfg_scale: float = 1.0) -> torch.Tensor:
        """
        Batched CFG like DhariwalUNet.forward_with_cfg:
          - Duplicate batch, null labels for second half inside this module,
            run a single forward, then combine uncond/cond.
        """
        if cfg_scale <= 1.0 or self.label_embed is None or y is None:
            return self.forward(x, t, y)  # no CFG

        B = x.shape[0]
        half = B // 2
        # Build 2*half batch: [cond_half, cond_half]
        x2 = torch.cat([x[:half], x[:half]], dim=0)
        t2 = torch.cat([t[:half], t[:half]], dim=0)
        y2 = torch.cat([y[:half], y[:half]], dim=0)

        # In the second half we will drop labels using drop_half_label=True
        v_all = self.forward(x2, t2, y2, drop_half_label=True)  # [2*half,...]
        v_cond, v_uncond = torch.split(v_all, half, dim=0)
        v_hat = v_uncond + cfg_scale * (v_cond - v_uncond)
        # Mirror shape back to B by repeating the guided half
        return torch.cat([v_hat, v_hat], dim=0)

####################################################
# Example usage in a reaction-diffusion context
class ReactionDiffusionWithGPS(nn.Module):
    """
    Example of using GPSDiffusion within the reaction-diffusion framework.
    Similar to ReactionDiffusion7_forNonLinearDiffusion2 but using our GPS model with authentic RWPE.
    """
    def __init__(
        self,
        base_flow_network,
        in_channels: int = 4,
        latent_size: int = 32,
        diffusion: bool = True,
        activation_type: str = 'ELU',
        time_channels: int = 64,
        time_hidden: int = 64,
        time_freqs: int = 8,
        hidden_dim: int = 64,
        pe_dim: int = 16,
        walk_length: int = 20,
        gps_layers: int = 5,
        heads: int = 4,
        attn_type: str = 'multihead',
        redraw_interval: int = 1000
    ):
        super().__init__()
        
        # Base flow network (reaction term)
        self.reaction = base_flow_network
        
        # Whether to use diffusion term
        self.diffusion = diffusion
        
        # Create GPS diffusion network if diffusion is enabled
        if self.diffusion:
            self.diffusion_term = GPSDiffusion(
                in_channels=in_channels,
                H=latent_size,
                W=latent_size,
                hidden_dim=hidden_dim,
                pe_dim=pe_dim,
                walk_length=walk_length,
                num_layers=gps_layers,
                heads=heads,
                attn_type=attn_type,
                time_channels=time_channels,
                time_hidden=time_hidden,
                time_freqs=time_freqs,
                redraw_interval=redraw_interval
            )
        
        # Activation function for combining terms (not used directly in forward pass)
        if activation_type == 'Softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation_type == 'SiLU':
            self.activation = nn.SiLU()
        elif activation_type == 'ReLU':
            self.activation = nn.ReLU()
        else:  # Default is ELU
            self.activation = nn.ELU()
    
    def forward(self, x, t):
        # Process through reaction network
        if self.reaction.__class__.__name__ == 'DiT' or self.reaction.__class__.__name__ == 'DhariwalUNet':
            reaction_term = self.reaction(t, x)  # DiT expects (t, x) ordering
        else:
            reaction_term = self.reaction(x, t)  # Most models expect (x, t) ordering
        
        # Apply diffusion if enabled
        if self.diffusion:
            # Process through GPS diffusion network
            diffusion_term = self.diffusion_term(x, t)
            
            # Combine reaction and diffusion terms (simple addition as in the hybrid_networks.py models)
            output = diffusion_term + reaction_term
            
            # Return combined output and metrics for monitoring
            return output, diffusion_term.abs().max(), reaction_term.abs().max()
        else:
            # If diffusion is disabled, just return reaction term
            return reaction_term, torch.tensor(1e-6, device=x.device), reaction_term.abs().max()


# Integration with hybrid_networks.py framework
def create_hybrid_model_with_gps(
    base_model, 
    in_channels=4, 
    hidden_dim=64, 
    model_type="diffusion_reaction_with_gps",
    diffusion=True, 
    activation_type='ELU', 
    time_channels=64, 
    time_hidden_channels=64, 
    time_num_frequencies=8,
    walk_length=20,
    gps_layers=5,
    heads=4,
    attn_type='multihead',
    **kwargs
):
    """
    Factory function to create a hybrid model with GPS diffusion.
    To be added to create_hybrid_model function in hybrid_networks.py
    """
    if model_type == 'diffusion_reaction_with_gps':
        return ReactionDiffusionWithGPS(
            base_flow_network=base_model,
            in_channels=in_channels,
            latent_size=32,  # Standard size for SD VAE latents
            diffusion=diffusion,
            activation_type=activation_type,
            time_channels=time_channels,
            time_hidden=time_hidden_channels,
            time_freqs=time_num_frequencies,
            hidden_dim=hidden_dim,
            pe_dim=16,
            walk_length=walk_length,
            gps_layers=gps_layers,
            heads=heads,
            attn_type=attn_type,
            redraw_interval=1000 if attn_type == 'performer' else None
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Example of integrating with your training script:
"""
# In your training script:
from networks.pnp_flow_UNet import UNet2
from networks.hybrid_networks import create_hybrid_model_with_gps

# Add to create_hybrid_model function in hybrid_networks.py:
elif model_type == 'diffusion_reaction_with_gps':
    return create_hybrid_model_with_gps(
        base_model=base_model,
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        model_type=model_type,
        diffusion=diffusion,
        activation_type=activation_type,
        time_channels=time_channels,
        time_hidden_channels=time_hidden_channels,
        time_num_frequencies=time_num_frequencies,
        walk_length=20,
        gps_layers=5,
        heads=4,
        attn_type='multihead'  # or 'performer'
    )

# Then in the training script:
vel_net = create_hybrid_model(
    base_model=base_model,
    in_channels=args.latent_channels,
    hidden_dim=64,
    model_type="diffusion_reaction_with_gps",  # new model type
    adj_mode=args.adj_mode,
    diffusion=args.diffusion,
    identity_Laplacian=args.identity_Laplacian,
    activation_type=args.Diffusion_term_activation_function,
    time_channels=args.time_channels,
    time_hidden_channels=args.time_hidden_channels,
    time_num_frequencies=args.time_num_frequencies
).to(args.device)

# Everything else in your training loop remains the same
"""