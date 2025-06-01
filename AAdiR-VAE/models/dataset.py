# =============================================================
# This code is provided by KetiveeAI for development purposes only.
# Not for sale or distribution. All rights reserved by KetiveeAI.
# See LICENSE for details.
# =============================================================

from transformers import BertTokenizer
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import torch
from torch.utils.data import Dataset

class TextImageDataset(Dataset):
    def __init__(self, image_urls, texts, transform=None, max_seq_length=64):
        self.image_urls = image_urls
        self.texts = texts
        self.transform = transform
        self.max_seq_length = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
    def __len__(self):
        return len(self.image_urls)
    
    def __getitem__(self, idx):
        # Load image
        image_url = self.image_urls[idx]
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
            
            # Tokenize text
            text = self.texts[idx]
            text_tokens = self.tokenizer(text, return_tensors="pt", padding='max_length', 
                                       truncation=True, max_length=self.max_seq_length)
            
            # Remove batch dimension from tokenizer output
            text_tokens = {k: v.squeeze(0) for k, v in text_tokens.items()}
            
            return image, text_tokens
        except:
            # Return a random item if there's an error
            return self[np.random.randint(0, len(self))]