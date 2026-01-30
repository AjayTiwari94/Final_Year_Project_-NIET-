import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *
from models.classifier import create_model
from utils.data_loader import create_dataloaders


def train_epoch(model, train_loader, criterion, optimizer, device, scaler):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def train_model(dataset_type, model_name='efficientnet', epochs=EPOCHS, resume=False):
    """
    Main training function
    
    Args:
        dataset_type: 'aptos', 'ham10000', or 'mura'
        model_name: 'efficientnet', 'resnet50', or 'densenet121'
        epochs: number of training epochs
        resume: whether to resume from checkpoint
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} on {dataset_type.upper()} dataset")
    print(f"{'='*60}\n")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(dataset_type, model_name)
    model = model.to(device)
    
    # Create data loaders
    print("\nLoading datasets...")
    train_loader, val_loader = create_dataloaders(dataset_type, BATCH_SIZE)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    scaler = GradScaler()
    
    # Training loop
    best_val_acc = 0.0
    start_epoch = 0
    
    # Resume from checkpoint if needed
    checkpoint_path = MODEL_DIR / f"{dataset_type}_{model_name}_best.pth"
    if resume and checkpoint_path.exists():
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
    
    print(f"\nStarting training from epoch {start_epoch}...\n")
    
    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Saved best model with Val Acc: {best_val_acc:.2f}%")
        
        print("=" * 60 + "\n")
    
    print(f"\nTraining completed! Best Val Acc: {best_val_acc:.2f}%")
    print(f"Model saved at: {checkpoint_path}")
    
    return model, best_val_acc


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train medical image classifier')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['aptos', 'ham10000', 'mura'],
                       help='Dataset to train on')
    parser.add_argument('--model', type=str, default='efficientnet',
                       choices=['efficientnet', 'resnet50', 'densenet121'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        dataset_type=args.dataset,
        model_name=args.model,
        epochs=args.epochs,
        resume=args.resume
    )
