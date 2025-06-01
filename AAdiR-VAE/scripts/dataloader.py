# =============================================================
# This code is provided by KetiveeAI for development purposes only.
# Not for sale or distribution. All rights reserved by KetiveeAI.
# See LICENSE for details.
# =============================================================


import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import BertTokenizer
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_seq_length=128):
        self.root_dir = root_dir
        self.transform = transform or self.default_transform()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_seq_length = max_seq_length

        # Look for captions.json
        captions_path = os.path.join(root_dir, 'captions.json')
        if not os.path.exists(captions_path):
            captions_path = os.path.join(os.path.dirname(root_dir), 'captions.json')

        if not os.path.exists(captions_path):
            raise FileNotFoundError(f"Could not find captions.json in {root_dir} or its parent directory")

        # Load captions
        with open(captions_path, 'r', encoding='utf-8') as f:
            self.captions = json.load(f)

        # Load all image paths that exist in captions.json
        self.image_paths = []
        for fname in self.captions.keys():
            full_path = os.path.join(root_dir, fname)
            if os.path.exists(full_path) and fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                self.image_paths.append(full_path)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No valid image files found in {root_dir} matching captions.json")

    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        filename = os.path.basename(img_path)
        caption = self.captions.get(filename, "")

        text = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length
        )

        # Return input_ids and attention_mask as tensors for training
        return {
            "input_ids": text["input_ids"].squeeze(0),
            "attention_mask": text["attention_mask"].squeeze(0),
            "image": image
        }
