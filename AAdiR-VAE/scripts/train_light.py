# =============================================================
# This code is provided by KetiveeAI for development purposes only.
# Not for sale or distribution. All rights reserved by KetiveeAI.
# See LICENSE for details.
# =============================================================
# This script trains the lightweight TinyVAE model for fast prototyping and low-resource environments.
# For high-quality/3D/video model training, see train.py (under development).
# =============================================================

# This script is designed to train a lightweight version of the AADI model using a subset of the dataset.
# This code is part of the AADI project, which is licensed under the MIT License.
# This code is provided for educational purposes only.
# Please ensure you have the necessary libraries installed:
# If you have any issues, please contact the author. A

import sys
import os
import yaml
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import random
import time
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.aadi_lite_vae import TinyVAE  # Updated import
from scripts.dataloader import CustomDataset
from utils.shared_state import shared_data
from scripts.utils.checkpoint_utils import save_checkpoint, load_checkpoint

# Logging setup
logging.basicConfig(
    filename='logs/train_lite.log',
    filemode='a',
    format='%(asctime)s %(message)s',
    level=logging.INFO
)

def log(msg):
    print(msg)
    logging.info(msg)

class LiteTrainer:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_epoch = 0

        # TensorBoard setup
        self.writer = SummaryWriter(log_dir='logs/tensorboard_lite')

        # Load dataset with limit
        full_dataset = CustomDataset(
            self.config['data']['dataset_path'],
            max_seq_length=self.config['data']['max_seq_length']
        )
        
        # Select random subset of images
        max_images = self.config['data'].get('max_images', 10000)
        if len(full_dataset) > max_images:
            indices = random.sample(range(len(full_dataset)), max_images)
            self.dataset = Subset(full_dataset, indices)
        else:
            self.dataset = full_dataset

        # Optimized dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )

        # Initialize model and optimizer
        self.model = TinyVAE(self.config['model']).to(self.device)  # Updated model
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate']
        )
        self.scaler = GradScaler(enabled=self.config['training']['use_amp'])

        # Try loading checkpoint
        resume_path = self.config['training'].get('resume_checkpoint')
        if resume_path and os.path.isfile(resume_path):
            self.start_epoch = load_checkpoint(self.model, self.optimizer, resume_path)
            log(f"✅ Resumed from checkpoint: {resume_path}")
        else:
            log("🆕 Starting fresh training")

    def load_config(self, path):
        with open(path) as f:
            return yaml.safe_load(f)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        
        grad_accum_steps = self.config['training']['gradient_accumulation_steps']
        num_batches = len(self.dataloader)
        
        start_time = time.time()
        for batch_idx, batch in enumerate(self.dataloader):
            if not shared_data.get("running", True):
                log("⏸️ Training paused/stopped!")
                break

            # Move data to device
            text = {
                "input_ids": batch["input_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device)
            }
            images = batch["image"].to(self.device)

            # Forward pass with mixed precision
            with autocast(device_type='cuda', enabled=self.config['training']['use_amp']):
                recon, mu, logvar = self.model(text, images)
                loss_dict = self.model.compute_loss(recon, images, mu, logvar)
                loss = loss_dict['loss'] / grad_accum_steps

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # Update metrics
            total_loss += loss.item() * grad_accum_steps
            total_recon_loss += loss_dict['recon_loss']
            total_kl_loss += loss_dict['kl_loss']

            # Log progress every 10 batches
            if batch_idx % 10 == 0:
                progress = (batch_idx + 1) / num_batches * 100
                curr_time = time.time() - start_time
                eta = curr_time / (batch_idx + 1) * (num_batches - batch_idx - 1)
                log(f"Epoch {epoch} [{progress:.1f}%] Loss: {loss.item():.4f} ETA: {eta/60:.1f}m")

                # Visualize current batch
                if batch_idx % 50 == 0:
                    self.writer.add_images('Original', images[:4], epoch * len(self.dataloader) + batch_idx)
                    self.writer.add_images('Reconstructed', recon[:4], epoch * len(self.dataloader) + batch_idx)

        # Calculate averages
        avg_loss = total_loss / len(self.dataloader)
        avg_recon = total_recon_loss / len(self.dataloader)
        avg_kl = total_kl_loss / len(self.dataloader)
        epoch_time = time.time() - start_time

        # Log to TensorBoard
        self.writer.add_scalar("Loss/Epoch", avg_loss, epoch)
        self.writer.add_scalar("Loss/Recon", avg_recon, epoch)
        self.writer.add_scalar("Loss/KL", avg_kl, epoch)
        self.writer.add_scalar("Time/Epoch", epoch_time, epoch)
        self.writer.flush()

        return avg_loss

    def train(self):
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            if not shared_data.get("running", True):
                log("⏸️ Training stopped!")
                break

            avg_loss = self.train_epoch(epoch)
            log(f"✅ Epoch {epoch} completed | Avg Loss: {avg_loss:.4f}")

            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                filename = f"checkpoints/lite_model_epoch_{epoch}.pth"
                save_checkpoint(self.model, self.optimizer, epoch, avg_loss, filename)
                log(f"💾 Saved checkpoint: {filename} (Loss: {avg_loss:.4f})")

        self.writer.close()

def launch_lite_training():
    shared_data["running"] = True
    trainer = LiteTrainer("configs/train_config_light.yaml")
    trainer.train()
    return "Lite training completed."

if __name__ == "__main__":
    launch_lite_training()