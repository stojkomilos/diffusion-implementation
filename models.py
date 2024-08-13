import torch.nn.functional as F
import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        #self.norm = nn.BatchNorm2d(in_channels)
        self.norm = nn.GroupNorm(compute_num_groups(in_channels), in_channels)
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=4, batch_first=True)
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

class TimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TimeEmbedding, self).__init__()
        self.d_model = d_model

    def forward(self, t):
        half_dim = self.d_model // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def compute_num_groups(num_channels, divisor=8):
    num_groups = max(1, num_channels // divisor)
    return num_groups

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.0):
        super(ResNetBlock, self).__init__()
        self.time_emb_lin = nn.Linear(time_emb_dim, out_channels)
        self.norm1 = nn.GroupNorm(compute_num_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(compute_num_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()
    
    def forward(self, x, t):
        time_emb = self.time_emb_lin(t).unsqueeze(-1).unsqueeze(-1)
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = h + time_emb
        h = self.dropout(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.dropout(h)

        return h + self.res_conv(x)

class UNet32(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.0):
        super(UNet32, self).__init__()
        self.time_emb = TimeEmbedding(time_emb_dim)
        
        self.enc1 = ResNetBlock(in_channels, 64, time_emb_dim, dropout)
        self.enc2 = ResNetBlock(64, 128, time_emb_dim, dropout)
        self.enc3 = ResNetBlock(128, 256, time_emb_dim, dropout)
        self.enc4 = ResNetBlock(256, 512, time_emb_dim, dropout)
        
        self.attn = AttentionBlock(128)
        
        self.bottleneck = ResNetBlock(512, 1024, time_emb_dim, dropout)
        
        self.dec4 = ResNetBlock(1024, 512, time_emb_dim, dropout)
        self.dec3 = ResNetBlock(512, 256, time_emb_dim, dropout)
        self.dec2 = ResNetBlock(256, 128, time_emb_dim, dropout)
        self.dec1 = ResNetBlock(128, 64, time_emb_dim, dropout)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x, t):
        t_emb = self.time_emb(t)
        
        enc1_out = self.enc1(x, t_emb)
        enc2_out = self.enc2(F.max_pool2d(enc1_out, 2), t_emb)
        enc3_out = self.enc3(F.max_pool2d(enc2_out, 2), t_emb)
        enc4_out = self.enc4(F.max_pool2d(enc3_out, 2), t_emb)
        
        attn_out = self.attn(enc2_out)
        
        bottleneck_out = self.bottleneck(F.max_pool2d(enc4_out, 2), t_emb)
        
        up4_out = self.up4(bottleneck_out)
        dec4_out = self.dec4(torch.cat([up4_out, enc4_out], dim=1), t_emb)
        
        up3_out = self.up3(dec4_out)
        dec3_out = self.dec3(torch.cat([up3_out, enc3_out], dim=1), t_emb)
        
        up2_out = self.up2(dec3_out)
        dec2_out = self.dec2(torch.cat([up2_out, attn_out], dim=1), t_emb)
        
        up1_out = self.up1(dec2_out)
        dec1_out = self.dec1(torch.cat([up1_out, enc1_out], dim=1), t_emb)
        
        out = self.final_conv(dec1_out)

        return out

class DiffusionModel(nn.Module):
    def __init__(self, grid_size, nr_channels, nr_hidden):
        super(DiffusionModel, self).__init__()

        self.grid_size = grid_size
        self.nr_channels = nr_channels

        self.fc1 = nn.Linear(in_features=grid_size*grid_size*nr_channels, out_features=nr_hidden)
        self.sigmoid1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=nr_hidden, out_features=grid_size*grid_size*nr_channels)

    def forward(self, x, t : torch.tensor):
        x = torch.view(-1, self.nr_channels, self.grid_size, self.grid_size)
        output = self.fc1(x)
        output = self.sigmoid1(output)
        output = self.fc2(output)

        output += x

        return output


class DiffusionModel(nn.Module):
    def __init__(self, grid_size, nr_channels, nr_hidden):
        super(DiffusionModel, self).__init__()

        self.grid_size = grid_size
        self.nr_channels = nr_channels

        self.fc1 = nn.Linear(in_features=grid_size*grid_size*nr_channels, out_features=nr_hidden)
        self.sigmoid1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=nr_hidden, out_features=grid_size*grid_size*nr_channels)

    def forward(self, x, t : torch.tensor):
        x = torch.view(-1, self.nr_channels, self.grid_size, self.grid_size)
        output = self.fc1(x)
        output = self.sigmoid1(output)
        output = self.fc2(output)

        output += x

        return output
    
if __name__ == "__main__":
    model = UNet32(in_channels=3, out_channels=1, time_emb_dim=128)
    x = torch.randn(8, 3, 32, 32)  # batch of 8 images
    t = torch.randn(8)  # batch of 8 time steps
    print(t.shape)
    out = model(x, t)
    print(out.shape)  # should be (8, 1, 32, 32)