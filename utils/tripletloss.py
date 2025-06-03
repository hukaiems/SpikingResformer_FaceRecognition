import torch
from torch import nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, embeddings, targets=None):
        # Handle both [T, 3*B, embed_dim] and [3*B, embed_dim] inputs
        if embeddings.dim() == 3:
            T, batch_size, embed_dim = embeddings.shape
        elif embeddings.dim() == 2:
            T = 1
            batch_size, embed_dim = embeddings.shape
            embeddings = embeddings.unsqueeze(0)  # Add time dimension: [1, 3*B, embed_dim]
        else:
            raise ValueError(f"Expected 2D or 3D input, got shape {embeddings.shape}")

        assert batch_size % 3 == 0, "Batch size must be divisible by 3 for triplets"
        
        # Reshape to separate triplet components
        embeddings = embeddings.reshape(T, batch_size//3, 3, embed_dim)
        
        # Extract anchor, positive, and negative
        anchor = embeddings[:, :, 0]    # [T, B/3, embed_dim]
        positive = embeddings[:, :, 1]  # [T, B/3, embed_dim]
        negative = embeddings[:, :, 2]  # [T, B/3, embed_dim]
        
        # Calculate distances
        pos_dist = torch.sum((anchor - positive) ** 2, dim=-1)  # [T, B/3]
        neg_dist = torch.sum((anchor - negative) ** 2, dim=-1)  # [T, B/3]
        
        # Triplet loss with margin
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0)
        
        # Mean over all dimensions
        loss = loss.mean()
        return loss

class OnlineTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # embeddings: [B, D]; labels: [B]
        # 1) pairwise distance
        dist = torch.cdist(embeddings, embeddings, p=2)              # [B, B]

        loss = 0.0
        n = 0
        for i in range(embeddings.size(0)):
            label_i = labels[i]
            # positives & negatives masks
            pos_mask = (labels == label_i)
            neg_mask = (labels != label_i)

            # hardest positive
            pos_dists = dist[i][pos_mask]
            pos_dists = pos_dists[pos_dists > 0]                      # exclude self-distance zero
            hardest_pos = pos_dists.max()

            # semi-hard negatives: dist > hardest_pos
            neg_dists = dist[i][neg_mask]
            semi = neg_dists[neg_dists > hardest_pos]
            if semi.numel() > 0:
                hardest_neg = semi.min()
            else:
                hardest_neg = neg_dists.min()

            triplet = F.relu(hardest_pos - hardest_neg + self.margin)
            loss += triplet
            n += 1

        return loss / n