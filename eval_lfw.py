import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import LFWPairs
import torch.nn.functional as F
from timm.models import create_model
from sklearn.metrics import roc_curve, auc, accuracy_score
import numpy as np
from spikingjelly.activation_based import functional

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate on LFW')
    parser.add_argument('--lfw-root', type=str, required=True,
                        help='Path to extracted lfw-funneled directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--model', type=str, default='spikingresformer_ti',
                        help='Model architecture name')
    parser.add_argument('--T', type=int, default=4,
                        help='Number of time steps')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--input-size', type=int, nargs=2, default=[224,224],
                        help='Model input size (H W)')
    parser.add_argument('--embed-dim', type=int, default=512,
                        help='Embedding dimension')
    return parser.parse_args()

def extract_embeddings(model, image, T):
    """Extract embeddings from the model, handling spiking neural network specifics"""
    model.eval()
    with torch.no_grad():
        # Forward pass through the model
        output = model(image)  # Shape: [T, B, embed_dim] for spiking models
        
        # Reset the spiking network state
        functional.reset_net(model)
        
        # Average over time steps if output has time dimension
        if len(output.shape) == 3 and output.shape[0] == T:
            embeddings = output.mean(0)  # [B, embed_dim]
        else:
            embeddings = output  # [B, embed_dim]
        
        # L2 normalize embeddings for better similarity computation
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
    return embeddings

def main():
    args = parse_args()
    
    # 1. Define transforms (same as training)
    val_transforms = transforms.Compose([
        transforms.Resize(tuple(args.input_size)),
        transforms.CenterCrop(tuple(args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])
    
    # 2. Load LFW pairs dataset
    dataset = LFWPairs(root=args.lfw_root,
                       split='test',
                       download=False,
                       transform=val_transforms)
    
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True)
    
    # 3. Model initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with embedding output (not classification)
    model = create_model(
        args.model,
        T=args.T,
        num_classes=args.embed_dim,  # Use embedding dimension as output
        img_size=args.input_size[0],
    ).cuda()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Handle different checkpoint structures
    if 'model' in ckpt:
        state_dict = ckpt['model']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    
    model.load_state_dict(state_dict)
    model.to(device).eval()
    
    print(f"Model loaded successfully. Evaluating on LFW...")
    print(f"Dataset size: {len(dataset)} pairs")
    
    # 4. Evaluate
    all_labels = []
    all_distances = []
    all_similarities = []
    
    with torch.no_grad():
        for batch_idx, ((img1, img2), labels) in enumerate(loader):
            img1, img2 = img1.to(device), img2.to(device)
            
            # Extract embeddings for both images
            emb1 = extract_embeddings(model, img1, args.T)
            emb2 = extract_embeddings(model, img2, args.T)
            
            # Compute distances and similarities
            distances = torch.norm(emb1 - emb2, p=2, dim=1).cpu().numpy()
            similarities = torch.sum(emb1 * emb2, dim=1).cpu().numpy()  # cosine similarity
            
            all_distances.append(distances)
            all_similarities.append(similarities)
            all_labels.append(labels.numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(loader)} batches")
    
    # Concatenate all results
    all_distances = np.concatenate(all_distances)
    all_similarities = np.concatenate(all_similarities)
    all_labels = np.concatenate(all_labels)
    
    print(f"\nTotal pairs evaluated: {len(all_labels)}")
    print(f"Positive pairs: {np.sum(all_labels)}")
    print(f"Negative pairs: {len(all_labels) - np.sum(all_labels)}")
    
    # 5. Compute metrics using distance (lower distance = more similar)
    fpr_dist, tpr_dist, thresholds_dist = roc_curve(all_labels, -all_distances)
    roc_auc_dist = auc(fpr_dist, tpr_dist)
    
    # 6. Compute metrics using similarity (higher similarity = more similar)
    fpr_sim, tpr_sim, thresholds_sim = roc_curve(all_labels, all_similarities)
    roc_auc_sim = auc(fpr_sim, tpr_sim)
    
    print(f"\n=== Results ===")
    print(f"Distance-based AUC: {roc_auc_dist:.4f}")
    print(f"Similarity-based AUC: {roc_auc_sim:.4f}")
    
    # Find best accuracy with distance threshold
    best_acc_dist = 0
    best_thresh_dist = 0
    for thr in thresholds_dist:
        preds = (all_distances < thr).astype(int)
        acc = accuracy_score(all_labels, preds)
        if acc > best_acc_dist:
            best_acc_dist, best_thresh_dist = acc, thr
    
    # Find best accuracy with similarity threshold
    best_acc_sim = 0
    best_thresh_sim = 0
    for thr in thresholds_sim:
        preds = (all_similarities > thr).astype(int)
        acc = accuracy_score(all_labels, preds)
        if acc > best_acc_sim:
            best_acc_sim, best_thresh_sim = acc, thr
    
    print(f"Best accuracy (distance): {best_acc_dist:.4f} at threshold {best_thresh_dist:.4f}")
    print(f"Best accuracy (similarity): {best_acc_sim:.4f} at threshold {best_thresh_sim:.4f}")
    
    # Summary statistics
    print(f"\n=== Distance Statistics ===")
    print(f"Mean distance (same): {all_distances[all_labels==1].mean():.4f}")
    print(f"Mean distance (different): {all_distances[all_labels==0].mean():.4f}")
    print(f"Std distance (same): {all_distances[all_labels==1].std():.4f}")
    print(f"Std distance (different): {all_distances[all_labels==0].std():.4f}")

if __name__ == '__main__':
    main()