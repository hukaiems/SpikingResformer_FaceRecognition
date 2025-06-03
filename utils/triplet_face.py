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
        try:
            with open(triplet_list_file, 'r') as f:
                for line in f:
                    anchor, pos, neg = line.strip().split()
                    self.triplets.append((anchor, pos, neg))
        except Exception as e:
            raise RuntimeError(f"Failed to read triplet list {triplet_list_file}: {e}")

        self.use_face_alignment = use_face_alignment
        self.target_size = target_size
        self.transform = transform or T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a_path, p_path, n_path = self.triplets[idx]
        
        if self.use_face_alignment:
            try:
                a = align_and_crop(a_path, self.target_size)
                p = align_and_crop(p_path, self.target_size)
                n = align_and_crop(n_path, self.target_size)
            except Exception as e:
                print(f"Face alignment failed for triplet {idx}: {e}")
                # Fallback to blank RGB images
                a = Image.new('RGB', self.target_size, (128, 128, 128))
                p = Image.new('RGB', self.target_size, (128, 128, 128))
                n = Image.new('RGB', self.target_size, (128, 128, 128))
        else:
            try:
                a = Image.open(a_path).convert('RGB')
                p = Image.open(p_path).convert('RGB')
                n = Image.open(n_path).convert('RGB')
            except Exception as e:
                print(f"Image loading failed for triplet {idx}: {e}")
                a = Image.new('RGB', self.target_size, (128, 128, 128))
                p = Image.new('RGB', self.target_size, (128, 128, 128))
                n = Image.new('RGB', self.target_size, (128, 128, 128))

        # Apply transforms
        a_t, p_t, n_t = self.transform(a), self.transform(p), self.transform(n)

        # Verify tensor shapes
        expected_shape = (3, self.target_size[0], self.target_size[1])
        for name, tensor in [("anchor", a_t), ("positive", p_t), ("negative", n_t)]:
            if tensor.shape != expected_shape:
                print(f"Shape mismatch for {name} at index {idx}: got {tensor.shape}, expected {expected_shape}")
                # Replace with dummy tensor to avoid crashing
                tensor = torch.zeros(expected_shape)

        return a_t, p_t, n_t