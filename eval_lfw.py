import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from timm.models import create_model
from sklearn.metrics import roc_curve, auc, accuracy_score
import numpy as np
from spikingjelly.activation_based import functional

class LFWPairsDataset(Dataset):
    """Custom LFW pairs dataset that works with standard LFW directory structure"""
    
    def __init__(self, lfw_root, transform=None, download_pairs=True):
        self.lfw_root = lfw_root
        self.transform = transform
        self.pairs = []
        
        # Download and create pairs file if needed
        pairs_file = os.path.join(lfw_root, 'pairs.txt')
        
        if not os.path.exists(pairs_file) and download_pairs:
            print("Downloading LFW pairs.txt file...")
            self._download_pairs_file(pairs_file)
        
        if not os.path.exists(pairs_file):
            print("Creating default pairs from available images...")
            self._create_default_pairs()
        else:
            print(f"Loading pairs from {pairs_file}")
            self._load_pairs_from_file(pairs_file)
            
        print(f"Loaded {len(self.pairs)} pairs")
    
    def _download_pairs_file(self, pairs_file):
        """Download the official LFW pairs.txt file"""
        try:
            import urllib.request
            url = "http://vis-www.cs.umass.edu/lfw/pairs.txt"
            urllib.request.urlretrieve(url, pairs_file)
            print(f"Downloaded pairs.txt to {pairs_file}")
        except Exception as e:
            print(f"Failed to download pairs.txt: {e}")
    
    def _load_pairs_from_file(self, pairs_file):
        """Load pairs from the official pairs.txt file"""
        with open(pairs_file, 'r') as f:
            lines = f.readlines()
        
        # Parse the file
        line_idx = 0
        while line_idx < len(lines):
            line = lines[line_idx].strip()
            if not line or line.startswith('#'):
                line_idx += 1
                continue
                
            parts = line.split()
            if len(parts) == 1 and parts[0].isdigit():
                # This is a header line indicating number of same pairs
                num_same_pairs = int(parts[0])
                line_idx += 1
                
                # Read same person pairs
                for _ in range(num_same_pairs):
                    if line_idx >= len(lines):
                        break
                    line = lines[line_idx].strip()
                    parts = line.split()
                    if len(parts) == 3:
                        name, idx1, idx2 = parts
                        img1_path = os.path.join(self.lfw_root, name, f"{name}_{idx1.zfill(4)}.jpg")
                        img2_path = os.path.join(self.lfw_root, name, f"{name}_{idx2.zfill(4)}.jpg")
                        if os.path.exists(img1_path) and os.path.exists(img2_path):
                            self.pairs.append((img1_path, img2_path, 1))
                    line_idx += 1
                
                # Read different person pairs (rest of the file)
                while line_idx < len(lines):
                    line = lines[line_idx].strip()
                    if line and not line.startswith('#'):
                        parts = line.split()
                        if len(parts) == 4:
                            name1, idx1, name2, idx2 = parts
                            img1_path = os.path.join(self.lfw_root, name1, f"{name1}_{idx1.zfill(4)}.jpg")
                            img2_path = os.path.join(self.lfw_root, name2, f"{name2}_{idx2.zfill(4)}.jpg")
                            if os.path.exists(img1_path) and os.path.exists(img2_path):
                                self.pairs.append((img1_path, img2_path, 0))
                    line_idx += 1
                break
            else:
                line_idx += 1
    
    def _create_default_pairs(self):
        """Create a default set of pairs from available images"""
        print("Creating default pairs from directory structure...")
        
        # Get all person directories
        person_dirs = []
        for item in os.listdir(self.lfw_root):
            person_path = os.path.join(self.lfw_root, item)
            if os.path.isdir(person_path):
                images = [f for f in os.listdir(person_path) if f.endswith('.jpg')]
                if len(images) >= 2:
                    person_dirs.append((item, images))
        
        # Create same person pairs
        same_pairs = 0
        for person, images in person_dirs:
            if len(images) >= 2:
                # Take first two images
                img1 = os.path.join(self.lfw_root, person, images[0])
                img2 = os.path.join(self.lfw_root, person, images[1])
                self.pairs.append((img1, img2, 1))
                same_pairs += 1
                if same_pairs >= 300:  # Limit to 300 same pairs
                    break
        
        # Create different person pairs
        diff_pairs = 0
        for i, (person1, images1) in enumerate(person_dirs):
            for j, (person2, images2) in enumerate(person_dirs[i+1:], i+1):
                img1 = os.path.join(self.lfw_root, person1, images1[0])
                img2 = os.path.join(self.lfw_root, person2, images2[0])
                self.pairs.append((img1, img2, 0))
                diff_pairs += 1
                if diff_pairs >= 300:  # Limit to 300 different pairs
                    break
            if diff_pairs >= 300:
                break
        
        print(f"Created {same_pairs} same person pairs and {diff_pairs} different person pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        
        # Load images
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
        except Exception as e:
            print(f"Error loading images: {img1_path}, {img2_path}")
            print(f"Error: {e}")
            # Return dummy images
            img1 = Image.new('RGB', (224, 224))
            img2 = Image.new('RGB', (224, 224))
        
        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return (img1, img2), label

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate on LFW')
    parser.add_argument('--lfw-root', type=str, required=True,
                        help='Path to LFW root directory (containing person folders)')
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
    
    print(f"LFW root directory: {args.lfw_root}")
    print(f"Checking directory structure...")
    
    # Check if the directory exists and has the right structure
    if not os.path.exists(args.lfw_root):
        print(f"Error: LFW root directory does not exist: {args.lfw_root}")
        return
    
    # List some directories to verify structure
    subdirs = [d for d in os.listdir(args.lfw_root) if os.path.isdir(os.path.join(args.lfw_root, d))]
    print(f"Found {len(subdirs)} person directories")
    if len(subdirs) > 0:
        print(f"Sample directories: {subdirs[:5]}")
    
    # 1. Define transforms (same as training)
    val_transforms = transforms.Compose([
        transforms.Resize(tuple(args.input_size)),
        transforms.CenterCrop(tuple(args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])
    
    # 2. Load LFW pairs dataset with custom implementation
    dataset = LFWPairsDataset(root=args.lfw_root, transform=val_transforms)
    
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True)
    
    # 3. Model initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model with embedding output (not classification)
    model = create_model(
        args.model,
        T=args.T,
        num_classes=args.embed_dim,  # Use embedding dimension as output
        img_size=args.input_size[0],
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
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