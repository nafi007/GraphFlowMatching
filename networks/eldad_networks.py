import torch
import torch.nn as nn
import torch.nn.functional as F


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

# ASPP Block
#------------------------
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=(1, 2, 4, 6)):
        super(ASPP, self).__init__()
        
        # 1x1 Convolution (no dilation)
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 Convolutions with different dilation rates
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

        # Final 1x1 Conv to fuse all branches
        self.conv1x1_out = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1x1_1(x)))  # 1x1 Conv
        x2 = F.relu(self.bn2(self.conv3x3_1(x)))  # 3x3 Conv (small dilation)
        x3 = F.relu(self.bn3(self.conv3x3_2(x)))  # 3x3 Conv (medium dilation)
        x4 = F.relu(self.bn4(self.conv3x3_3(x)))  # 3x3 Conv (large dilation)
        
        # Concatenate all features
        x = torch.cat([x1, x2, x3, x4], dim=1)

        # Final fusion
        x = self.conv1x1_out(x)
        return x

         
# ------------------------
# Resnet Block
# ------------------------
class resnet(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels, num_layers=3):
        super(resnet, self).__init__()
        self.num_layers = num_layers
        self.OpenConv  = ASPP(in_channels, hid_channels, dilations=(1, 2, 4, 6))
        self.CloseConv = ASPP(hid_channels, out_channels, dilations=(1, 2, 4, 6))
        
        self.layers = nn.ModuleList()
        self.te     = nn.ModuleList()
        for i in range(num_layers):
            Bi = ASPP(hid_channels, hid_channels, dilations=(1, 2, 4, 6))
            self.layers.append(Bi)
            
            Ti = TimeEmbedding(hid_channels, hidden_dim=hid_channels, num_freqs=8)  
            self.te.append(Ti)
            
    def forward(self, x, t):
        x = self.OpenConv(x)
        for i in range(self.num_layers):
            te = self.te[i](t).unsqueeze(-1).unsqueeze(-1)
            x = x + self.layers[i](x + te)
        x = self.CloseConv(x)
        return x

# ------------------------
# Encoder-Decoder with Adaptive Layers
# ------------------------
class Encoder(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, num_layers=3, H=32, latent_channels=64):
        super().__init__()
        self.num_layers = num_layers
        
        self.OpenConv = nn.Conv2d(in_channels, hid_channels, kernel_size=3, padding=1)
        self.layers = nn.ParameterList()
        self.ChannelUp = nn.ModuleList()
        for i in range(num_layers):
            Bi = ASPP(hid_channels, 2*hid_channels, dilations=(1, 2, 4, 6))
            self.layers.append(Bi)
            self.ChannelUp.append(nn.Conv2d(hid_channels, hid_channels*2, kernel_size=1))
            H = H // 2
            hid_channels = 2*hid_channels
            
        self.fcMu  = nn.Conv2d(hid_channels, latent_channels, kernel_size=3, padding=1)
        self.fcVar = nn.Conv2d(hid_channels, latent_channels, kernel_size=3, padding=1)    
        self.fcVar.bias.data.fill_(-2.0)
        self.hid_channels = hid_channels
        self.H    = H
        
    def forward(self, x):
        x = self.OpenConv(x)
        for i in range(self.num_layers):
            x = self.ChannelUp[i](x) + self.layers[i](x)
            x = F.avg_pool2d(x, 2)
        
        mu = self.fcMu(x)
        log_var = self.fcVar(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


# ------------------------
# Decoder with Adaptive Layers
# ------------------------
class Decoder(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, num_layers=3, latent_channels=64, H=64):
        super().__init__()
        self.num_layers = num_layers
        self.H = H
        self.hid_channels = hid_channels
        self.ConvMu = nn.Conv2d(latent_channels, hid_channels, kernel_size=3, padding=1)
        
        self.layers = nn.ParameterList()
        self.ChannelDown = nn.ModuleList()
        for i in range(num_layers):
            Bi = ASPP(hid_channels, hid_channels//2, dilations=(1, 2, 4, 6))
            self.layers.append(Bi) 
            self.ChannelDown.append(nn.Conv2d(hid_channels, hid_channels//2, kernel_size=1)) 
            H = H * 2
            hid_channels = hid_channels//2
        self.CloseConv = nn.Conv2d(hid_channels, in_channels, kernel_size=1)
            
    def forward(self, x):
        x = self.ConvMu(x)
        for i in range(self.num_layers):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            dx = self.layers[i](x)
            x = self.ChannelDown[i](x) + dx
            
        x = self.CloseConv(x)
        return x
