# =============================================================
# This code is provided by KetiveeAI for development purposes only.
# Not for sale or distribution. All rights reserved by KetiveeAI.
# See LICENSE for details.
# =============================================================

import torch
import os
from scripts.utils import save_checkpoint

def save_checkpoint(model, optimizer, filename):
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return True
    return False