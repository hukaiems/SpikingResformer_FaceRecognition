# datasets/triplet_face.py
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class TripletFaceDataset(Dataset):
    def __init__(self, triplet_list_file, transform=None):
        """
        triplet_list_file: a .txt where each line is:
          /path/to/anchor.jpg /path/to/positive.jpg /path/to/negative.jpg
        """
        self.triplets = []
        with open(triplet_list_file, 'r') as f:
            for line in f:
                anchor, pos, neg = line.strip().split()
                self.triplets.append((anchor, pos, neg))
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
        return (
            self.transform(a),
            self.transform(p),
            self.transform(n),
        )
