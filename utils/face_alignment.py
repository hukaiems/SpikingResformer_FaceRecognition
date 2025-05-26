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
    if aligned is None or not isinstance(aligned, torch.Tensor):
        print("Warning: MTCNN failed, fallback to original image")
        return img.resize(target_size)
    if aligned.is_cuda:
        aligned = aligned.cpu()
        aligned = aligned.permute(1, 2, 0).mul(255).byte().numpy()
        aligned_pil = Image.fromarray(aligned)
    if target_size != (160, 160):
        aligned_pil = aligned_pil.resize(target_size)
    return aligned_pil