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
        
        if self.use_face_alignment:
            try:
                # Pass file paths directly to align_and_crop
                a = align_and_crop(a_path)
                p = align_and_crop(p_path)
                n = align_and_crop(n_path)
                
                    
            except Exception as e:
                # If face alignment fails, fall back to original images
                print(f"Face alignment failed for triplet {idx}: {e}")
                a = Image.open(a_path).convert('RGB')
                p = Image.open(p_path).convert('RGB')
                n = Image.open(n_path).convert('RGB')
        else:
            # Load images normally without face alignment
            a = Image.open(a_path).convert('RGB')
            p = Image.open(p_path).convert('RGB')
            n = Image.open(n_path).convert('RGB')

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
        
        if self.use_face_alignment:
            try:
                # Pass file paths directly to align_and_crop
                a = align_and_crop(a_path)
                p = align_and_crop(p_path)
                n = align_and_crop(n_path)
                
                    
            except Exception as e:
                # If face alignment fails, fall back to original images
                print(f"Face alignment failed for triplet {idx}: {e}")
                a = Image.open(a_path).convert('RGB')
                p = Image.open(p_path).convert('RGB')
                n = Image.open(n_path).convert('RGB')
        else:
            # Load images normally without face alignment
            a = Image.open(a_path).convert('RGB')
            p = Image.open(p_path).convert('RGB')
            n = Image.open(n_path).convert('RGB')

        a_t, p_t, n_t = self.transform(a), self.transform(p), self.transform(n)
        print(f"Anchor stats: min {a_t.min()}, max {a_t.max()}, mean {a_t.mean()}")
        return a_t, p_t, n_t