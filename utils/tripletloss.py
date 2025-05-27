import torch
from torch import nn

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