# =============================================================
# This code is provided by KetiveeAI for development purposes only.
# Not for sale or distribution. All rights reserved by KetiveeAI.
# See LICENSE for details.
# =============================================================
# rvae_handler.py
# This script handles the RVAE model for image generation.
# plese ensure you have the necessary libraries installed:
# Don't sell or distribute this code without permission.
# All rights reserved by KetiveeAI. 
import sys
import os

import torch

# Add the parent folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

from aadir_vae import AadiR_VAE

model = AadiR_VAE().cuda()

def generate_image(text: str, image_tensor):
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer(text, return_tensors="pt").to("cuda")
    image_tensor = image_tensor.cuda()
    with torch.no_grad():
        output, _, _ = model(tokens, image_tensor)
    return output
