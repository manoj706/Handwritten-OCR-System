import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from typing import List, Tuple
from .utils import LabelEncoder

class HandwritingDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[str], label_encoder: LabelEncoder, img_height: int = 32, img_width: int = 128, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.label_encoder = label_encoder
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
             # Create a valid placeholder image (black) if load fails
            image = np.zeros((self.img_height, self.img_width), dtype=np.uint8)

        # Resize and Pad
        h, w = image.shape
        # Resize height to self.img_height, scale width roughly
        scale = self.img_height / h
        new_w = int(w * scale)
        image = cv2.resize(image, (new_w, self.img_height))
        
        # Pad or Crop width
        if new_w < self.img_width:
            # Pad
            padded = np.ones((self.img_height, self.img_width), dtype=np.uint8) * 255 # White background
            padded[:, :new_w] = image
            image = padded
        else:
            # Crop
            image = image[:, :self.img_width]

        # Normalize to [0, 1] and add channel dim
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0) # (1, H, W)

        # Encode label
        encoded_label = self.label_encoder.encode(label)
        encoded_label = torch.LongTensor(encoded_label)
        label_len = torch.LongTensor([len(encoded_label)])

        if self.transform:
            image = self.transform(image)
        else:
             image = torch.FloatTensor(image)

        return image, encoded_label, label_len

def handwriting_collate_fn(batch):
    images, encoded_labels, label_lens = zip(*batch)
    
    # Stack images
    images = torch.stack(images, dim=0)
    
    # Concatenate all labels into a single 1D tensor
    targets = torch.cat(encoded_labels)
    
    # Stack label lengths
    target_lengths = torch.cat(label_lens)
    
    return images, targets, target_lengths
