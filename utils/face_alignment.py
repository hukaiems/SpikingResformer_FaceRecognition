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
    """
    1) Detects the largest face
    2) Aligns landmarks so eye-nose-mouth are canonical
    3) Returns a cropped, squared PIL image at target_size
    
    Args:
        img_input: Either a file path (str) or PIL Image
        target_size: Tuple of (width, height) for output size
    """
    # Handle both file path and PIL Image inputs
    if isinstance(img_input, str):
        img = Image.open(img_input).convert('RGB')
    else:
        img = img_input.convert('RGB')
    
    # Use MTCNN for face detection and alignment
    mtcnn = _get_mtcnn()
    aligned = mtcnn(img)           # returns a torch.Tensor or None
    if aligned is None:
        # fallback: center crop + resize
        return img.resize(target_size)
    
    # Convert back to PIL
    aligned_pil = Image.fromarray(aligned.permute(1,2,0).int().numpy())
    
    # Resize to target size if different from MTCNN's default (160x160)
    if target_size != (160, 160):
        aligned_pil = aligned_pil.resize(target_size)
    
    return aligned_pil