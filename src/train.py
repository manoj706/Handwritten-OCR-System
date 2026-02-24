import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import time

from src.dataset import HandwritingDataset, handwriting_collate_fn
from src.model import CRNN
from src.utils import LabelEncoder, compute_cer, compute_wer

def train(args):
    # Devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Load Labels
    with open(args.labels_path, "r") as f:
        lines = f.readlines()
    
    image_paths = []
    labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            filename = parts[0]
            label = " ".join(parts[1:])
            path = os.path.join(os.path.dirname(args.labels_path), filename)
            if os.path.exists(path):
                image_paths.append(path)
                labels.append(label)
    
    print(f"Found {len(image_paths)} images.")
    
    # Label Encoder
    all_chars = "".join(labels)
    label_encoder = LabelEncoder(all_chars)
    print(f"Vocab size: {len(label_encoder.characters)}")
    
    # Save vocab
    with open(os.path.join(args.save_dir, "vocab.txt"), "w") as f:
        f.write("".join(label_encoder.characters))
    
    # Dataset & Loader
    dataset = HandwritingDataset(image_paths, labels, label_encoder, img_height=32, img_width=128)
    
    # Split
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=handwriting_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=handwriting_collate_fn)
    
    # Model
    num_classes = len(label_encoder.characters) + 1 # +1 for Blank
    model = CRNN(img_height=32, num_classes=num_classes).to(device)
    
    # Loss & Optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_cer = float('inf')
    
    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for images, targets, target_lengths in pbar:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            # Forward
            preds = model(images) # (B, W, C)
            
            # (W, B, C) for CTCLoss
            preds_permuted = preds.permute(1, 0, 2) 
            
            # Log Softmax
            preds_log_softmax = nn.functional.log_softmax(preds_permuted, dim=2)
            
            # Input Lengths: (B,)
            # We assume the output width is fixed for the fixed input width
            # CNN reduces width by 4 (pool 2, pool 2).
            # So 128 -> 32.
            # But wait, my CNN maxpool breakdown:
            # 128 -> 64 -> 32 -> 16 (pool3 2,1) -> 8 (pool4 2,1) -> 8 (conv6 stride 1)
            # Wait, let's trace W:
            # W=128 -> Pool1(2,2) -> 64
            # 64 -> Pool2(2,2) -> 32
            # 32 -> Pool3(2,1) -> 32 (stride w is 1)
            # 32 -> Pool4(2,1) -> 32 (stride w is 1)
            # 32 -> Conv6 -> 32
            # So W_out = 32 roughly.
            # Let's verify input_lengths.
            
            feature_w = preds.size(1) 
            input_lengths = torch.full(size=(images.size(0),), fill_value=feature_w, dtype=torch.long).to(device)
            
            loss = criterion(preds_log_softmax, targets, input_lengths, target_lengths)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        # Validation
        val_cer, val_wer = evaluate(model, val_loader, label_encoder, device)
        print(f"Epoch {epoch+1} - Train Loss: {train_loss/len(train_loader):.4f} - Val CER: {val_cer:.4f} - Val WER: {val_wer:.4f}")
        
        # Save checkpoint
        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))
            print(f"Saved best model with CER: {val_cer:.4f}")

def evaluate(model, loader, encoder, device):
    model.eval()
    total_cer = 0
    total_wer = 0
    num_samples = 0
    
    with torch.no_grad():
        for images, targets, target_lengths in loader:
            images = images.to(device)
            
            preds = model(images) # (B, W, C)
            
            # Decode predictions
            decoded_preds = encoder.decode_greedy(preds)
            
            # Decode targets
            # targets is 1D concatenated. target_lengths tells us how to split it.
            # We need to split targets back to list of target strings.
            start = 0
            ts = targets.cpu().numpy()
            tls = target_lengths.cpu().numpy()
            
            decoded_targets = []
            for length in tls:
                t = ts[start:start+length]
                # t are indices. 0 is blank? My LabelEncoder says 0 is blank.
                # LabelEncoder.decode removes blank (0).
                # But targets from dataset shouldn't have blanks ideally, just chars.
                decoded_targets.append(encoder.decode(t, remove_blank=False))
                start += length
            
            # Compute Metrics
            for p, t in zip(decoded_preds, decoded_targets):
                total_cer += compute_cer(p, t)
                total_wer += compute_wer(p, t)
                num_samples += 1
                
    if num_samples == 0:
        return 0.0, 0.0
        
    return total_cer / num_samples, total_wer / num_samples

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_path", type=str, required=True, help="Path to labels.txt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_dir", type=str, default=".")
    args = parser.parse_args()
    
    train(args)
