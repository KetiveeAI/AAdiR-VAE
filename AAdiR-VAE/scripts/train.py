# =============================================================
# This code is provided by KetiveeAI for development purposes only.
# Not for sale or distribution. All rights reserved by KetiveeAI.
# See LICENSE for details.
# =============================================================
# This training script contains two model architectures:
# 1. High-Quality/3D/Video Model (AadiR_VAE): For advanced, high-resolution image and video generation (under development).
# 2. Lightweight Model: See train_light.py for fast prototyping and low-resource environments.
# =============================================================

import sys
import os
import glob
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import torchvision
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.aadir_vae import AadiR_VAE
from scripts.dataloader import CustomDataset
from utils.shared_state import shared_data
from scripts.utils.checkpoint_utils import save_checkpoint, load_checkpoint

import logging

# Logging to file
logging.basicConfig(
    filename='logs/train.log',
    filemode='a',
    format='%(asctime)s %(message)s',
    level=logging.INFO
)
def log(msg):
    print(msg)
    logging.info(msg)


class Trainer:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_epoch = 0

        if torch.cuda.is_available():
            # Set PyTorch memory allocation settings
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # TensorBoard setup
        self.writer = SummaryWriter(log_dir='logs/tensorboard')

        # Load dataset and dataloader
        self.dataset = CustomDataset(
            self.config['data']['dataset_path'],
            max_seq_length=self.config['data']['max_seq_length']
        )
        # Optimize dataloader settings
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=shared_data.get("batch_size", self.config['training']['batch_size']),
            shuffle=True,
            num_workers=2,  
            pin_memory=True,  # Re-enable pin_memory for faster GPU transfer
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=2  # Add prefetching to reduce data loading overhead
        )

        # Init model and optimizer
        self.model = AadiR_VAE(self.config['model']).to(self.device)
        
        # Set VGG to use gradient checkpointing if available
        if hasattr(self.model, '_vgg') and self.model._vgg is not None:
            self.model._vgg.use_gradient_checkpointing = True
        
        # Use a larger initial learning rate with warmup
        self.optimizer = Adam(self.model.parameters(), lr=self.config['training']['learning_rate'] * 2)
        self.kl_weight = self.config['training'].get('kl_weight', 1e-5)
        self.scaler = GradScaler(enabled=True)  # Explicitly enable mixed precision

        # Try to load checkpoint
        resume_path = self.config['training'].get('resume_checkpoint', None)
        if resume_path and os.path.isfile(resume_path):
            self.start_epoch = load_checkpoint(self.model, self.optimizer, resume_path)
            log(f"✅ Resumed from checkpoint: {resume_path} (Epoch {self.start_epoch})")
        else:
            log("🆕 Starting training from scratch.")

    def load_config(self, path):
        with open(path) as f:
            config = yaml.safe_load(f)
            
            # Override batch size from environment if set
            if "FORCE_BATCH_SIZE" in os.environ:
                forced_batch_size = int(os.environ["FORCE_BATCH_SIZE"])
                print(f"\nOverriding batch size from environment: {forced_batch_size}")
                config['training']['batch_size'] = forced_batch_size
                
            # Ensure proper float conversion            config['training']['learning_rate'] = float(config['training']['learning_rate'])
            config['training']['kl_weight'] = float(config['training'].get('kl_weight', 1e-5))
            
            # Enforce memory optimization settings
            config['training']['use_amp'] = True
            config['training']['memory_efficient'] = True
            config['model']['use_checkpoint'] = True
            
            return config

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        total_perceptual_loss = 0.0
        grad_accum_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        num_batches = len(self.dataloader)

        # Zero gradients at the start of accumulation
        self.optimizer.zero_grad()
        
        start_time = time.time()
        batch_times = []
          # Grid for image visualization
        def make_grid(images, nrow=4):
            # Normalize images to [0,1] range first
            images = (images + 1) / 2
            return torchvision.utils.make_grid(images, nrow=nrow, normalize=False)
        
        for batch_idx, batch in enumerate(self.dataloader):
            if not shared_data.get("running", True):
                log(f"⏸️ Training paused/stopped!")
                break

            batch_start = time.time()
            try:
                # Move data to device
                text = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device)
                }
                images = batch["image"].to(self.device)

                # Forward pass with mixed precision
                with autocast('cuda', enabled=True):
                    recon, mu, logvar = self.model(text, images)
                    loss_dict = self.model.compute_loss(recon, images, mu, logvar)
                    
                    # Force finite loss
                    if not torch.isfinite(loss_dict['loss']):
                        print(f"⚠️ [Batch {batch_idx}] Non-finite loss detected - skipping batch")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                        self.optimizer.zero_grad()
                        continue
                        
                    loss = loss_dict['loss'] / grad_accum_steps

                # Visualize images every 50 batches
                if batch_idx % 50 == 0:
                    with torch.no_grad():
                        # Original images
                        img_grid = make_grid(images[:8])
                        self.writer.add_image('Images/Original', img_grid, epoch * len(self.dataloader) + batch_idx)
                        
                        # Reconstructed images
                        recon_grid = make_grid(recon[:8])
                        self.writer.add_image('Images/Reconstructed', recon_grid, epoch * len(self.dataloader) + batch_idx)
                        
                        # Sample new images from random latents
                        num_samples = 8
                        rand_z = torch.randn(num_samples, self.model.latent_dim).to(self.device)
                        text_output = self.model.text_encoder(**text)
                        text_emb = self.model.text_fc(text_output.last_hidden_state[:1, 0, :])  # Take first sequence, CLS token
                        # Properly expand text embedding to match batch size
                        rand_text_emb = text_emb.expand(num_samples, -1)  # Expand to match batch size, keeping features
                        
                        # Calculate base feature size
                        base_h = self.model.feature_h
                        base_w = self.model.feature_w
                        
                        # Create dummy features with correct sizes matching encoder output
                        dummy_feats = [
                            torch.zeros(num_samples, 64, base_h * 8, base_w * 8).to(self.device),  # First skip connection
                            torch.zeros(num_samples, 128, base_h * 4, base_w * 4).to(self.device), # Second skip connection
                            torch.zeros(num_samples, 256, base_h * 2, base_w * 2).to(self.device), # Third skip connection
                            torch.zeros(num_samples, 512, base_h, base_w).to(self.device),         # Fourth skip connection
                        ]
                        samples = self.model.decode(rand_z, rand_text_emb, dummy_feats)
                        sample_grid = make_grid(samples)
                        self.writer.add_image('Images/Sampled', sample_grid, epoch * len(self.dataloader) + batch_idx)
                        
                        self.writer.flush()

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % grad_accum_steps == 0:
                    # Unscale before clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Update with scaled gradients
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                # Update metrics
                total_loss += loss.item() * grad_accum_steps
                total_recon_loss += loss_dict['recon_loss']
                total_kl_loss += loss_dict['kl_loss']
                total_perceptual_loss += loss_dict.get('perceptual_loss', 0.0)

                # Calculate timing
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                avg_batch_time = sum(batch_times[-10:]) / min(len(batch_times), 10)
                eta = avg_batch_time * (num_batches - batch_idx)
                
                # Log progress with timing info 
                if batch_idx % 10 == 0:  # Reduced logging frequency from every batch to every 10 batches
                    progress = (batch_idx + 1) / num_batches * 100
                    progress_bar = "=" * int(progress/2) + ">" + " " * (50 - int(progress/2))
                    log(f"Epoch: {epoch} [{progress_bar}] {progress:.1f}%")
                    log(f"Batch: {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
                    log(f"Speed: {batch_time:.2f}s/batch, ETA: {eta/60:.1f}m")
                    
                    # Write batch loss and timing to TensorBoard
                    step = epoch * len(self.dataloader) + batch_idx
                    self.writer.add_scalar("Loss/Batch", loss.item(), step)
                    self.writer.add_scalar("Time/BatchProcessing", batch_time, step)
                    self.writer.flush()
                
                # Clear GPU cache less frequently
                if batch_idx % 200 == 0:  # Increased from 100 to 200
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    log(f"❌ CUDA out of memory in batch {batch_idx}. Skipping batch.")
                    continue
                else:
                    raise e

        # Log timing info for epoch
        epoch_time = time.time() - start_time
        avg_batch_time = sum(batch_times) / len(batch_times)
        log(f"\nEpoch {epoch} timing:")
        log(f"- Total time: {epoch_time/60:.2f}m")
        log(f"- Average batch time: {avg_batch_time:.2f}s")
        
        # Calculate average losses for the epoch
        avg_loss = total_loss / len(self.dataloader)
        avg_recon = total_recon_loss / len(self.dataloader)
        avg_kl = total_kl_loss / len(self.dataloader)
        avg_perceptual = total_perceptual_loss / len(self.dataloader)

        # Log epoch averages
        self.writer.add_scalar("Loss/Epoch", avg_loss, epoch)
        self.writer.add_scalar("Loss/Epoch_Reconstruction", avg_recon, epoch)
        self.writer.add_scalar("Loss/Epoch_KL", avg_kl, epoch)
        self.writer.add_scalar("Loss/Epoch_Perceptual", avg_perceptual, epoch)
        self.writer.add_scalar("Time/Epoch", epoch_time, epoch)
        self.writer.add_scalar("Time/AverageBatch", avg_batch_time, epoch)
        self.writer.flush()
        
        return avg_loss

    def train(self):
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            if not shared_data.get("running", True):
                log(f"⏸️ Training stopped before epoch: {epoch}")
                break

            avg_loss = self.train_epoch(epoch)
            log(f"✅ Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                save_path = f"checkpoints/model_epoch_{epoch}.pth"
                save_checkpoint(self.model, self.optimizer, epoch, save_path)
                log(f"💾 Saved checkpoint: {save_path}")

        self.writer.close()


def launch_training():
    shared_data["running"] = True
    trainer = Trainer("configs/train_config.yaml")
    trainer.train()
    return "Training completed."


if __name__ == "__main__":
    shared_data["running"] = True
    trainer = Trainer("configs/train_config.yaml")
    trainer.train()
