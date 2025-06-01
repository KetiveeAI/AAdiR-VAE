# -*- coding: utf-8 -*-
# =============================================================
# This code is provided by KetiveeAI for development purposes only.
# Not for sale or distribution. All rights reserved by KetiveeAI.
# See LICENSE for details.
# =============================================================
# This file contains the high-quality/3D/video AadiR_VAE model architecture (under development).
# =============================================================

# AadiR VAE Model  
# This model is a Variational Autoencoder (VAE) that combines image and text encoders.
# It uses a convolutional neural network for image encoding and BERT for text encoding.
# Author : KetiveeAI team all rights reserved.

import torch
import torch.nn as nn
from transformers import BertModel
from torchvision.models import vgg16, VGG16_Weights
import torch.nn.functional as F
from torch.amp import autocast

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(B, -1, H * W)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=2)
        value = self.value(x).view(B, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x

class TriplanarFeatures(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_xy = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv_xz = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv_yz = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        # Simulate triplanar projection by processing features in 3 orthogonal planes
        feat_xy = self.act(self.norm(self.conv_xy(x)))
        feat_xz = self.act(self.norm(self.conv_xz(x)))
        feat_yz = self.act(self.norm(self.conv_yz(x)))
        return (feat_xy + feat_xz + feat_yz) / 3.0

class AadiR_VAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Extract parameters from config
        self.image_size = config.get('image_size', 64)
        self.image_channels = config.get('image_channels', 3)
        self.text_embed_dim = config.get('text_embed_dim', 768)
        self.latent_dim = config.get('latent_dim', 128)  # Reduced from 256
        
        # Enable gradient checkpointing
        self.use_checkpoint = True
        
        # Calculate feature dimensions
        self.feature_h = self.image_size // 32
        self.feature_w = self.image_size // 32
        
        # Enable torch.cuda memory optimizations
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
            
        # Memory-efficient encoder with reduced channels
        self.encoder = nn.ModuleList([
            self.conv_block(self.image_channels, 32),    # Reduced from 64
            self.conv_block(32, 64),                     # Reduced from 128
            self.conv_block(64, 128),                    # Reduced from 256
            self.conv_block(128, 256),                   # Reduced from 384
            self.conv_block(256, 512, norm=True)
        ])
        
        # Reduced feature extraction sizes
        self.triplanar_blocks = nn.ModuleList([
            TriplanarFeatures(64, 64),
            TriplanarFeatures(128, 128),
            TriplanarFeatures(256, 256),
            TriplanarFeatures(512, 512)
        ])

        # Reduced attention sizes
        self.attention_blocks = nn.ModuleList([
            SelfAttention(64),
            SelfAttention(128),
            SelfAttention(256),
            SelfAttention(512)
        ])

        # Calculate flattened dimension
        self.flatten_dim = 512 * self.feature_h * self.feature_w
        
        # Improved latent space mapping with smaller dimensions
        self.fc_mu = nn.Sequential(
            nn.Linear(self.flatten_dim, self.flatten_dim // 4),  # Reduced intermediate size
            nn.LeakyReLU(0.2),
            nn.Linear(self.flatten_dim // 4, self.latent_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(self.flatten_dim, self.flatten_dim // 4),  # Reduced intermediate size
            nn.LeakyReLU(0.2),
            nn.Linear(self.flatten_dim // 4, self.latent_dim)
        )

        # Lazy load VGG
        self._vgg = None
        
        # Text Encoder with reduced projection
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.text_fc = nn.Sequential(
            nn.Linear(self.text_embed_dim, 256),  # Reduced from 512
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.latent_dim)
        )
        self.text_encoder.requires_grad_(False)

        # Enhanced decoder input with reduced dimensions
        self.decoder_input = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.flatten_dim),
            nn.LayerNorm(self.flatten_dim),
            nn.LeakyReLU(0.2)
        )

        # Improved decoder blocks with reduced channels
        self.up1 = self.res_deconv_block(1024, 256)   # Reduced from 384
        self.up2 = self.res_deconv_block(512, 128)    # Reduced from 256
        self.up3 = self.res_deconv_block(256, 64)     # Reduced from 128
        self.up4 = self.res_deconv_block(128, 32)     # Reduced from 64
        self.up5 = self.res_deconv_block(32, 16)      # Reduced from 32

        # Simplified final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, self.image_channels, 3, 1, 1),
            nn.Sigmoid()
        )

    @property
    def vgg(self):
        """Lazy load VGG only when needed"""
        if self._vgg is None:
            self._vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
            for param in self._vgg.parameters():
                param.requires_grad = False
            self._vgg.eval()
            if torch.cuda.is_available():
                self._vgg = self._vgg.cuda()
        return self._vgg

    def conv_block(self, in_channels, out_channels, norm=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        if norm:
            layers.insert(1, nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

    def res_deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def encode(self, x):
        feats = []
        for i, (enc_block, tri_block, attn_block) in enumerate(zip(
            self.encoder, 
            [None] + list(self.triplanar_blocks), 
            [None] + list(self.attention_blocks)
        )):
            # Only checkpoint if input requires grad and block has parameters
            if self.use_checkpoint and self.training and x.requires_grad and any(p.requires_grad for p in enc_block.parameters()):
                x = torch.utils.checkpoint.checkpoint(enc_block, x, use_reentrant=False)
            else:
                x = enc_block(x)
            if i > 0:
                if self.use_checkpoint and self.training and x.requires_grad and any(p.requires_grad for p in tri_block.parameters()):
                    x = torch.utils.checkpoint.checkpoint(tri_block, x, use_reentrant=False)
                else:
                    x = tri_block(x)
                if self.use_checkpoint and self.training and x.requires_grad and any(p.requires_grad for p in attn_block.parameters()):
                    x = torch.utils.checkpoint.checkpoint(attn_block, x, use_reentrant=False)
                else:
                    x = attn_block(x)
            feats.append(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar, feats

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, text_emb, feats):
        z_cat = torch.cat((z, text_emb), dim=1)
        x = self.decoder_input(z_cat)
        x = x.view(x.size(0), 512, self.feature_h, self.feature_w)
        skip4 = feats[-1]
        if self.use_checkpoint and self.training and x.requires_grad and any(p.requires_grad for p in self.up1.parameters()):
            x = torch.utils.checkpoint.checkpoint(lambda a, b: self.up1(torch.cat([a, b], dim=1)), x, skip4, use_reentrant=False)
        else:
            x = self.up1(torch.cat([x, skip4], dim=1))
        skip3 = feats[-2]
        if self.use_checkpoint and self.training and x.requires_grad and any(p.requires_grad for p in self.up2.parameters()):
            x = torch.utils.checkpoint.checkpoint(lambda a, b: self.up2(torch.cat([a, b], dim=1)), x, skip3, use_reentrant=False)
        else:
            x = self.up2(torch.cat([x, skip3], dim=1))
        skip2 = feats[-3]
        if self.use_checkpoint and self.training and x.requires_grad and any(p.requires_grad for p in self.up3.parameters()):
            x = torch.utils.checkpoint.checkpoint(lambda a, b: self.up3(torch.cat([a, b], dim=1)), x, skip2, use_reentrant=False)
        else:
            x = self.up3(torch.cat([x, skip2], dim=1))
        skip1 = feats[-4]
        if self.use_checkpoint and self.training and x.requires_grad and any(p.requires_grad for p in self.up4.parameters()):
            x = torch.utils.checkpoint.checkpoint(lambda a, b: self.up4(torch.cat([a, b], dim=1)), x, skip1, use_reentrant=False)
        else:
            x = self.up4(torch.cat([x, skip1], dim=1))
        if self.use_checkpoint and self.training and x.requires_grad and any(p.requires_grad for p in self.up5.parameters()):
            x = torch.utils.checkpoint.checkpoint(self.up5, x, use_reentrant=False)
        else:
            x = self.up5(x)
        x = self.final_conv(x)
        return x

    def reconstruction_loss(self, x_recon, x):
        """Enhanced reconstruction loss with L1 and perceptual components"""
        # L1 loss for sharp details
        l1_loss = F.l1_loss(x_recon, x, reduction='mean')
        
        # MSE loss for overall structure
        mse_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # Perceptual loss for high-level features
        perceptual = self.perceptual_loss(x_recon, x)
        
        # Combined loss with weights
        return 0.5 * l1_loss + 0.3 * mse_loss + 0.2 * perceptual

    def kl_loss(self, mu, logvar):
        """KL divergence loss with annealing factor and logvar clamping for stability"""
        # Clamp logvar to avoid exp overflow
        logvar = torch.clamp(logvar, min=-10, max=10)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        beta = getattr(self, 'beta', 1.0)
        return beta * kld.mean()

    def style_consistency_loss(self, z1, z2):
        """Style consistency loss between paired latent codes"""
        return F.mse_loss(z1, z2, reduction='mean')

    def _denorm_for_vgg(self, x):
        # If input is in [-1, 1], convert to [0, 1]
        return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

    def perceptual_loss(self, x_recon, x):
        """Calculate perceptual loss using VGG features, with input denormalization and clamping"""
        x_recon_vgg = self._denorm_for_vgg(x_recon)
        x_vgg = self._denorm_for_vgg(x)
        vgg_recon = self.vgg(x_recon_vgg)
        vgg_real = self.vgg(x_vgg)
        if not torch.isfinite(vgg_recon).all():
            print('[NaN/Inf] vgg_recon (perceptual_loss)')
        if not torch.isfinite(vgg_real).all():
            print('[NaN/Inf] vgg_real (perceptual_loss)')
        return F.mse_loss(vgg_recon, vgg_real, reduction='mean')

    def texture_loss(self, x_recon, x):
        """Stable Gram matrix loss with NaN/Inf prevention"""
        def safe_gram_matrix(feat):
            b, c, h, w = feat.size()
            feat = feat.view(b, c, h * w)
            
            # Normalize features to prevent overflow
            feat = feat / (torch.norm(feat, dim=(1, 2), keepdim=True) + 1e-8)
            
            # Stable Gram matrix calculation
            gram = torch.bmm(feat, feat.transpose(1, 2))
            return gram / (c * h * w + 1e-8)  # Safe division
        
        # Detach VGG and use float32 for stability
        with torch.no_grad(), autocast(device_type='cuda', enabled=False):
            x_recon_vgg = self._denorm_for_vgg(x_recon).float()
            x_vgg = self._denorm_for_vgg(x).float()
            vgg_recon = self.vgg(x_recon_vgg)
            vgg_real = self.vgg(x_vgg)
        
        # Double-check for NaNs before Gram matrix
        if not torch.isfinite(vgg_recon).all() or not torch.isfinite(vgg_real).all():
            return torch.tensor(0.0, device=x.device)  # Skip if invalid
        
        gram_recon = safe_gram_matrix(vgg_recon)
        gram_real = safe_gram_matrix(vgg_real)
        
        return F.mse_loss(gram_recon, gram_real)

    def compute_loss(self, x_recon, x, mu, logvar, z=None, z_pair=None):
        """Compute total loss with robust handling of unstable components"""
        recon_loss = self.reconstruction_loss(x_recon, x)
        kl_div = self.kl_loss(mu, logvar)
        
        # Clamp texture loss to prevent explosion
        tex_loss = torch.nan_to_num(
            self.texture_loss(x_recon, x),
            nan=0.0, posinf=1.0, neginf=0.0
        )
        
        # Style consistency if paired samples available
        style_loss = 0.0
        if z is not None and z_pair is not None:
            style_loss = self.style_consistency_loss(z, z_pair)
        
        # Combine all losses with safety checks
        total_loss = recon_loss + kl_div + 0.1 * tex_loss + 0.05 * style_loss
        
        # Improved NaN/Inf checks with detailed reporting
        components = {
            'recon_loss': recon_loss,
            'kl_loss': kl_div,
            'texture_loss': tex_loss,
            'style_loss': style_loss,
            'total_loss': total_loss
        }
        
        for name, val in components.items():
            if isinstance(val, torch.Tensor) and not torch.isfinite(val).all():
                print(f"[NaN/Inf] {name}: {val.item() if val.numel() == 1 else 'tensor'}")
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_div.item(),
            'texture_loss': tex_loss.item(),
            'style_loss': style_loss if isinstance(style_loss, float) else style_loss.item()
        }

    def set_beta(self, beta):
        """Set KL annealing factor"""
        self.beta = beta

    def forward(self, text_input, image, image_pair=None):
        """Enhanced forward pass with optional style consistency"""
        with torch.no_grad():
            text_outputs = self.text_encoder(**text_input)
        text_emb = self.text_fc(text_outputs.last_hidden_state[:, 0, :])

        # Encode main image
        mu, logvar, feats = self.encode(image)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, text_emb, feats)
        
        # Return reconstruction and latent variables
        return x_recon, mu, logvar

    @torch.no_grad()
    def sample(self, text_input, num_samples=1, temperature=1.0):
        """Generate samples with temperature control"""
        # Get text embedding
        text_outputs = self.text_encoder(**text_input)
        text_emb = self.text_fc(text_outputs.last_hidden_state[:, 0, :])
        
        # Repeat text embedding for multiple samples by expanding dimensions properly
        text_emb = text_emb.unsqueeze(0).repeat(num_samples, 1) if num_samples > 1 else text_emb
        
        # Sample from latent space with temperature
        z = torch.randn(num_samples, self.latent_dim).to(text_emb.device)
        z = z * temperature
        
        # Create dummy features matching encoder dimensions
        dummy_feats = [
            torch.zeros(num_samples, 512, self.feature_h, self.feature_w).to(z.device),
            torch.zeros(num_samples, 256, self.feature_h * 2, self.feature_w * 2).to(z.device),
            torch.zeros(num_samples, 128, self.feature_h * 4, self.feature_w * 4).to(z.device),
            torch.zeros(num_samples, 64, self.feature_h * 8, self.feature_w * 8).to(z.device),
        ]
        
        return self.decode(z, text_emb, dummy_feats)