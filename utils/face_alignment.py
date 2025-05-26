# utils/face_alignment.py
from facenet_pytorch import MTCNN
from PIL import Image
from typing import Union
import torch

# Global variable to store MTCNN instance per process
_mtcnn = None

def _get_mtcnn():
    """Get MTCNN instance, creating one per process if needed"""
    global _mtcnn
    if _mtcnn is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
    return _mtcnn

def align_and_crop(img_input: Union[str, Image.Image], target_size=(160, 160)) -> Image.Image:
    # Handle both file path and PIL Image inputs
    if isinstance(img_input, str):
        img = Image.open(img_input).convert('RGB')
    else:
        img = img_input.convert('RGB')
    
    # Use MTCNN for face detection and alignment
    mtcnn = _get_mtcnn()
    aligned = mtcnn(img)  # returns a torch.Tensor or None
    if aligned is None:
        # Fallback: resize original image
        return img.resize(target_size)
    
    # Check if the aligned image is too small
    if aligned.shape[1] < 10 or aligned.shape[2] < 10:
        print(f"Warning: Aligned image too small: {aligned.shape}, falling back to original")
        return img.resize(target_size)
    
    # Convert to PIL with correct dtype (uint8)
    aligned_pil = Image.fromarray(aligned.permute(1, 2, 0).byte().numpy())
    
    # Resize to target size if different from MTCNN's default (160x160)
    if target_size != (160, 160):
        aligned_pil = aligned_pil.resize(target_size)
    
    return aligned_pil