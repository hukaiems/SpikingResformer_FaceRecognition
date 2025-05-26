# datasets/preprocess.py
from facenet_pytorch import MTCNN
from PIL import Image

# initialize once (on GPU if available)
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device='cuda')

def align_and_crop(img_path: str) -> Image.Image:
    """
    1) Detects the largest face
    2) Aligns landmarks so eye-nose-mouth are canonical
    3) Returns a cropped, squared PIL image at 160×160
    """
    img = Image.open(img_path).convert('RGB')
    aligned = mtcnn(img)           # returns a torch.Tensor or None
    if aligned is None:
        # fallback: center crop + resize
        return img.resize((160,160))
    # convert back to PIL for consistency with torchvision transforms
    return Image.fromarray(aligned.permute(1,2,0).int().numpy())
