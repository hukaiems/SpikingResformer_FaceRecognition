# datasets/triplet_face.py
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from utils.face_alignment import align_and_crop 

class TripletFaceDataset(Dataset):
    def __init__(self, triplet_list_file, transform=None, use_face_alignment=True, target_size=(224, 224)):
        """
        triplet_list_file: a .txt where each line is:
          /path/to/anchor.jpg /path/to/positive.jpg /path/to/negative.jpg
        use_face_alignment: Whether to apply face alignment before transforms
        target_size: Target size for face alignment (width, height)
        """
        self.triplets = []
        with open(triplet_list_file, 'r') as f:
            for line in f:
                anchor, pos, neg = line.strip().split()
                self.triplets.append((anchor, pos, neg))

        self.use_face_alignment = use_face_alignment
        self.target_size = target_size

        self.transform = transform or T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225]),
        ])

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a_path, p_path, n_path = self.triplets[idx]
        a = Image.open(a_path).convert('RGB')
        p = Image.open(p_path).convert('RGB')
        n = Image.open(n_path).convert('RGB')

        if self.use_face_alignment:
            try:
                a = align_and_crop(a, target_size=self.target_size)
                p = align_and_crop(p, target_size=self.target_size)
                n = align_and_crop(n, target_size=self.target_size)
            except Exception as e:
                # If face alignment fails, fall back to original images
                print(f"Face alignment failed for triplet {idx}: {e}")
                # Keep original images and let transforms handle resizing

        return (
            self.transform(a),
            self.transform(p),
            self.transform(n),
        )
