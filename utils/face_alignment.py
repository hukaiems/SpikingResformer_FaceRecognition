# utils/face_alignment.py
from facenet_pytorch import MTCNN
from PIL import Image
from typing import Union
import torch

_mtcnn = None

def _get_mtcnn():
    """Get MTCNN instance with strict configuration"""
    global _mtcnn
    if _mtcnn is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _mtcnn = MTCNN(
            image_size=160, 
            margin=30,
            keep_all=False, 
            min_face_size=50,
            thresholds=[0.8, 0.9, 0.9],
            device=device
        )
    return _mtcnn

def align_and_crop(img_input: Union[str, Image.Image], target_size=(224, 224)) -> Image.Image:
    # Handle both file path and PIL Image inputs
    if isinstance(img_input, str):
        try:
            img = Image.open(img_input).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_input}: {e}")
            # Create a blank RGB image as fallback
            img = Image.new('RGB', target_size, (128, 128, 128))
    else:
        img = img_input.convert('RGB')
    
    # Use MTCNN for face detection and alignment
    mtcnn = _get_mtcnn()
    try:
        aligned = mtcnn(img)  # Returns a torch.Tensor or None
    except Exception as e:
        print(f"MTCNN failed for image: {e}")
        aligned = None

    if aligned is None or not isinstance(aligned, torch.Tensor):
        print("Warning: MTCNN failed, falling back to resized image")
        return img.resize(target_size, Image.BILINEAR)
    
    # Convert tensor to PIL Image
    aligned = aligned.cpu().permute(1, 2, 0).mul(255).byte().numpy()
    aligned_pil = Image.fromarray(aligned).convert('RGB')  # Ensure RGB
    return aligned_pil.resize(target_size, Image.BILINEAR)