# =============================================================
# This code is provided by KetiveeAI for development purposes only.
# Not for sale or distribution. All rights reserved by KetiveeAI.
# See LICENSE for details.
# =============================================================
# This file contains the lightweight TinyVAE model architecture for fast prototyping and low-resource environments.
# =============================================================

import torch
import torch.nn as nn
from transformers import BertModel

class TinyVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Fixed parameters for simplicity
        self.image_size = 128  # Fixed size
        self.image_channels = 3
        self.text_embed_dim = 768
        self.latent_dim = 32   # Very small latent space
        
        # Encoder - Simple and fixed architecture
        self.encoder = nn.Sequential(
            # 128x128 -> 64x64
            nn.Conv2d(self.image_channels, 16, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x64 -> 32x32
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32 -> 16x16
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16 -> 8x8
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Fixed dimensions for 8x8 feature maps
        self.feature_dim = 64 * 8 * 8  # 4096
        
        # Latent space projections
        self.fc_mu = nn.Linear(self.feature_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, self.latent_dim)
        
        # Text encoder (frozen)
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.text_fc = nn.Sequential(
            nn.Linear(self.text_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim)
        )
        self.text_encoder.requires_grad_(False)
        
        # Decoder
        self.decoder_input = nn.Linear(self.latent_dim * 2, self.feature_dim)
        
        self.decoder = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(inplace=True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(inplace=True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            # 64x64 -> 128x128
            nn.ConvTranspose2d(16, self.image_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x):
        # Ensure input size
        if x.size(-1) != self.image_size or x.size(-2) != self.image_size:
            x = nn.functional.interpolate(x, size=(self.image_size, self.image_size))
        
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten: batch_size x 4096
        return self.fc_mu(x), self.fc_logvar(x)

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(x.size(0), 64, 8, 8)  # Reshape to 8x8 feature maps
        return self.decoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, text, images):
        # Process text
        with torch.no_grad():
            text_features = self.text_encoder(**text).last_hidden_state[:, 0]
        text_latent = self.text_fc(text_features)
        
        # Process image
        mu, logvar = self.encode(images)
        z = self.reparameterize(mu, logvar)
        
        # Combine latents
        combined_latent = torch.cat([z, text_latent], dim=1)
        
        # Decode
        recon = self.decode(combined_latent)
        return recon, mu, logvar

    def compute_loss(self, recon, images, mu, logvar):
        # Ensure images are same size as reconstruction
        if images.size(-1) != self.image_size or images.size(-2) != self.image_size:
            images = nn.functional.interpolate(images, size=(self.image_size, self.image_size))
        
        # Simple reconstruction loss
        recon_loss = nn.functional.mse_loss(recon, images)
        
        # Reduced KL divergence weight
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
        
        # Total loss with minimal KL weight
        loss = recon_loss + 0.0001 * kl_loss
        
        return {
            'loss': loss,
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item()
        }
