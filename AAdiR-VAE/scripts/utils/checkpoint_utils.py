# =============================================================
# This code is provided by KetiveeAI for development purposes only.
# Not for sale or distribution. All rights reserved by KetiveeAI.
# See LICENSE for details.
# =============================================================

import torch
import os

def save_checkpoint(model, optimizer, epoch, loss, filename):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }, filename)

def map_keys(old_dict):
    new_dict = {}
    for old_key, value in old_dict.items():
        new_key = old_key

        # Handle encoder layers
        if old_key.startswith('encoder'):
            parts = old_key.split('.')
            if len(parts) >= 3:
                layer_idx = int(parts[1])
                new_key = f"enc{layer_idx+1}"
                remaining = '.'.join(parts[2:])
                new_key = f"{new_key}.{remaining}"

        # Handle decoder layers
        elif old_key.startswith('decoder'):
            parts = old_key.split('.')
            if len(parts) >= 3:
                layer_idx = int(parts[1])
                if layer_idx == 5:  # Handle final layer separately
                    new_key = f"final_up.{parts[2]}"
                else:
                    new_key = f"up{layer_idx+1}"
                    remaining = '.'.join(parts[2:])
                    new_key = f"{new_key}.{remaining}"

        new_dict[new_key] = value
    return new_dict

def load_checkpoint(model, optimizer, filename):
    print(f"Loading checkpoint: {filename}")
    checkpoint = torch.load(filename)
    
    if 'state_dict' in checkpoint:
        mapped_state_dict = map_keys(checkpoint['state_dict'])
        print("Mapped state dict keys. Attempting to load...")
        # Debug: Print first few keys before and after mapping
        print("\nFirst few original keys:", list(checkpoint['state_dict'].keys())[:5])
        print("\nFirst few mapped keys:", list(mapped_state_dict.keys())[:5])
        
        try:
            model.load_state_dict(mapped_state_dict, strict=False)
            print("Successfully loaded model state with key mapping")
        except Exception as e:
            print(f"Warning: Error loading mapped state dict: {e}")
            print("Attempting to load original state dict with strict=False...")
            model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    if optimizer and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
    
    epoch = checkpoint.get('epoch', 0)
    print(f"Loaded checkpoint from epoch {epoch}")
    return epoch
