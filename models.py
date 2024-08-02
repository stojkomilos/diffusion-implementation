import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class TimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TimeEmbedding, self).__init__()
        self.d_model = d_model

    def forward(self, t):
        return torch.stack([torch.sin(t / (10000 ** (2 * i / self.d_model))) if i % 2 == 0 else torch.cos(t / (10000 ** (2 * i / self.d_model))) for i in range(self.d_model)], dim=-1)


def compute_num_groups(num_channels, divisor=8):
    num_groups = max(1, num_channels // divisor)
    return num_groups


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(ResNetBlock, self).__init__()
        self.time_emb_lin = nn.Linear(time_emb_dim, out_channels)
        self.norm1 = nn.GroupNorm(compute_num_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(compute_num_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()
    
    def forward(self, x, t):
        time_emb = self.time_emb_lin(t).unsqueeze(-1).unsqueeze(-1)
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = x + time_emb
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=1, batch_first=True)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        norm_x = self.norm(x)
        
        qkv = self.qkv(norm_x)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(batch_size, channels, height * width).permute(0, 2, 1)
        k = k.reshape(batch_size, channels, height * width).permute(0, 2, 1)
        v = v.reshape(batch_size, channels, height * width).permute(0, 2, 1)

        attn_output, _ = self.attn(q, k, v)
        attn_output = attn_output.permute(0, 2, 1).reshape(batch_size, channels, height, width)

        proj_output = self.proj_out(attn_output)

        return proj_output
    
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super(UNet, self).__init__()
        self.time_emb = TimeEmbedding(time_emb_dim)
        
        self.enc1 = ResNetBlock(in_channels, 64, time_emb_dim)
        self.enc2 = ResNetBlock(64, 128, time_emb_dim)
        self.enc3 = ResNetBlock(128, 256, time_emb_dim)
        
        self.attn = AttentionBlock(128)
        
        self.bottleneck = ResNetBlock(256, 512, time_emb_dim)
        
        self.dec3 = ResNetBlock(512, 256, time_emb_dim)
        self.dec2 = ResNetBlock(256, 128, time_emb_dim)
        self.dec1 = ResNetBlock(128, 64, time_emb_dim)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x, t):
        t_emb = self.time_emb(t)
        
        enc1_out = self.enc1(x, t_emb)
        enc2_out = self.enc2(F.max_pool2d(enc1_out, 2), t_emb)
        enc3_out = self.enc3(F.max_pool2d(enc2_out, 2), t_emb)
        
        attn_out = self.attn(enc2_out)
        
        bottleneck_out = self.bottleneck(F.max_pool2d(enc3_out, 2), t_emb)
        
        up3_out = self.up3(bottleneck_out)
        dec3_out = self.dec3(torch.cat([up3_out, enc3_out], dim=1), t_emb)
        
        up2_out = self.up2(dec3_out)
        dec2_out = self.dec2(torch.cat([up2_out, attn_out], dim=1), t_emb)
        
        up1_out = self.up1(dec2_out)
        dec1_out = self.dec1(torch.cat([up1_out, enc1_out], dim=1), t_emb)
        
        out = self.final_conv(dec1_out)
        
        return out

"""
class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        gs = my_dataset.grid_size
        nc = my_dataset.nr_channels
        
        nr_hidden = 3*nc if not my_dataset.USE_EXISTING_DATASET else 800
        self.fc1 = nn.Linear(in_features=gs*gs*nc, out_features=nr_hidden)

        self.sigmoid1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=nr_hidden, out_features=gs*gs*nc)

    def forward(self, x):
        output = self.fc1(x)
        output = self.sigmoid1(output)
        output = self.fc2(output)

        output += x

        return output
"""
    
    
if __name__ == '__main__':
    # Check for UNet
    model = UNet(in_channels=1, out_channels=1, time_emb_dim=128)
    x = torch.randn(1, 1, 256, 256)
    t = torch.tensor([10.0])
    output = model(x, t)

    print(output.shape)