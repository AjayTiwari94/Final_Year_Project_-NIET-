import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import cv2
from torchvision import transforms
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *

class APTOSDataset(Dataset):
    """Dataset for APTOS Diabetic Retinopathy"""
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['id_code']
        img_path = self.img_dir / f"{img_id}.png"
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        label = int(self.df.iloc[idx]['diagnosis'])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class HAM10000Dataset(Dataset):
    """Dataset for HAM10000 Skin Lesions"""
    def __init__(self, csv_file, img_dirs, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dirs = [Path(d) for d in img_dirs]
        self.transform = transform
        
        # Create label mapping
        self.label_map = {label: idx for idx, label in enumerate(HAM10000_CLASSES.keys())}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['image_id']
        
        # Try to find image in both directories
        img_path = None
        for img_dir in self.img_dirs:
            path = img_dir / f"{img_id}.jpg"
            if path.exists():
                img_path = path
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Image {img_id} not found")
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        label = self.label_map[self.df.iloc[idx]['dx']]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class MURADataset(Dataset):
    """Dataset for MURA Bone X-rays"""
    def __init__(self, csv_file, base_dir, transform=None):
        self.df = pd.read_csv(csv_file, names=['path', 'label'])
        self.base_dir = Path(base_dir)
        self.transform = transform
        
        # Build image paths list
        self.image_paths = []
        self.labels = []
        
        for idx, row in self.df.iterrows():
            folder_path = self.base_dir / row['path']
            if folder_path.exists():
                # Get all images in the folder
                images = list(folder_path.glob('*.png'))
                for img_path in images:
                    self.image_paths.append(img_path)
                    self.labels.append(int(row['label']))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(augment=True):
    """Get image transforms for training and validation"""
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/Test transforms
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_dataloaders(dataset_type, batch_size=BATCH_SIZE):
    """Create train and validation dataloaders for each dataset"""
    
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)
    
    if dataset_type == 'aptos':
        train_dataset = APTOSDataset(
            csv_file=APTOS_DIR / 'train_1.csv',
            img_dir=APTOS_DIR / 'train_images',
            transform=train_transform
        )
        val_dataset = APTOSDataset(
            csv_file=APTOS_DIR / 'valid.csv',
            img_dir=APTOS_DIR / 'val_images',
            transform=val_transform
        )
    
    elif dataset_type == 'ham10000':
        train_dataset = HAM10000Dataset(
            csv_file=HAM10000_DIR / 'HAM10000_metadata.csv',
            img_dirs=[
                HAM10000_DIR / 'HAM10000_images_part_1',
                HAM10000_DIR / 'HAM10000_images_part_2'
            ],
            transform=train_transform
        )
        # Split for validation (80-20 split)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    elif dataset_type == 'mura':
        train_dataset = MURADataset(
            csv_file=MURA_DIR / 'train_labeled_studies.csv',
            base_dir=MURA_DIR,
            transform=train_transform
        )
        val_dataset = MURADataset(
            csv_file=MURA_DIR / 'valid_labeled_studies.csv',
            base_dir=MURA_DIR,
            transform=val_transform
        )
    
    else:
        raise ValueError(f"Dataset type {dataset_type} not supported")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing data loaders...")
    
    for dataset in ['aptos', 'ham10000', 'mura']:
        print(f"\n{dataset.upper()} Dataset:")
        try:
            train_loader, val_loader = create_dataloaders(dataset, batch_size=8)
            print(f"Train batches: {len(train_loader)}")
            print(f"Val batches: {len(val_loader)}")
            
            # Test one batch
            images, labels = next(iter(train_loader))
            print(f"Batch shape: {images.shape}")
            print(f"Labels: {labels[:5]}")
        except Exception as e:
            print(f"Error: {e}")
