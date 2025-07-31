import torch
from torch import nn
import torch.nn.fuctional as F
import math

class ArcFaceLoss(nn.Module):
    """
    Implements ArcFace loss for face recognition

    Args:
        embed_dim (int): Dimension of embedding vectors.
        num_classes (int): Number of classes.
        s (float): scale factor for logits.
        m (float): Angular margin
    
    Returns:
        torch.Tensor: Scalar loss value    
    """

    def __init__(self, embed_dim, num_classes, s=30.0, m=0.5):
        super().__init__()
        self.s = s  # Scale factor
        self.m = m  # Margin
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.weight = nn.Parameter(torch.Tensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = self.sin_m * m

    def forward(self, embeddings, labels):
        # Handle different input shapes
        if embeddings.dim() == 3:  # [T, B, embed_dim] - from TET
            # Take mean across time steps
            embeddings = embeddings.mean(0)  # [B, embed_dim]
        elif embeddings.dim() == 2:  # [B, embed_dim] - normal case
            pass
        else:
            raise ValueError(f"Unexpected embeddings shape: {embeddings.shape}")
        
        # # Debug prints
        # print(f"ArcFace input embeddings shape: {embeddings.shape}")
        # print(f"ArcFace labels shape: {labels.shape}")
        # print(f"Weight shape: {self.weight.shape}")
        
        # Normalize embeddings and weights
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)  # [B, embed_dim]
        weight_norm = F.normalize(self.weight, p=2, dim=1)     # [num_classes, embed_dim]
        
        # Compute cosine similarity
        cos_theta = F.linear(embeddings_norm, weight_norm)  # [B, num_classes]
        
        # Compute sin_theta
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2) + 1e-7)
        
        # Apply margin
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        
        # Create mask for target classes
        mask = torch.zeros_like(cos_theta).scatter_(1, labels.unsqueeze(1), 1.0)
        
        # Apply margin only to correct class
        output = cos_theta * (1.0 - mask) + cos_theta_m * mask
        output = output * self.s
        
        # Compute cross entropy loss
        loss = F.cross_entropy(output, labels)
        return loss




class TETArcFaceLoss(nn.Module):
    """TET-aware ArcFace loss that handles temporal outputs properly"""
    def __init__(self, embed_dim, num_classes, s=30.0, m=0.5, TET_phi=1.0, TET_lambda=0.0):
        super().__init__()
        self.arcface = ArcFaceLoss(embed_dim, num_classes, s, m)
        self.TET_phi = TET_phi
        self.TET_lambda = TET_lambda
        
    def forward(self, embeddings, labels):
        if embeddings.dim() == 3:  # [T, B, embed_dim]
            T, B, D = embeddings.shape
            total_loss = 0.0
            
            # Apply ArcFace loss at each time step
            for t in range(T):
                step_loss = self.arcface(embeddings[t], labels)  # [B, embed_dim]
                total_loss += (1.0 - self.TET_lambda) * step_loss
            
            # Add TET regularization if needed
            if self.TET_phi > 0:
                # Mean squared error between consecutive time steps
                mse_loss = 0.0
                for t in range(1, T):
                    mse_loss += F.mse_loss(embeddings[t], embeddings[t-1])
                total_loss += self.TET_phi * self.TET_lambda * mse_loss / (T - 1)
            
            return total_loss / T
        else:
            # Standard case
            return self.arcface(embeddings, labels)