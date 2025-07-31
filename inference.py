import torch
from torchvision import transforms
import yaml
from PIL import Image
import argparse
from timm.models import create_model
import numpy as np

def load_config(config_path):
    "Load YAML config file"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_image(img_path, input_size):
    transform = transform.Compose([
        transforms.Resize(input_size[-2:]), #take the last two elements
        transforms.CenterCrop(input_size[-2:]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0) # Add batch dimension

def main():
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file")
    parser.add_argument('img', type=str, required=True, help='Path to image for inference')
    args = parser.parse_args()

    # 1 Load config
    cfg = load_config(args.config)
    input_size = cfg.get('input_size', [3, 112, 112])
    model_name = cfg.get('model', 'spikingresformet_ti')
    checkpoint_path = cfg.get('checkpoint', './logs/checkpoint_best.pth')
    T = cfg.get('T', 4)

    # 2 Load model
    model = create_model(model_name, T=T, num_classes=cfg.get('embed_dim', 512), img_size=input_size[-1])
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    # 3 Preprocess image
    img_tensor = preprocess_image(args.img, input_size).cuda()

    # 4 Inference 
    with torch.no_grad():
        embedding = model(img_tensor)
        if embedding.dim() == 3:
            embedding = embedding.mean(0)
        embedding = embedding.squeeze(0).cpu().numpy
    
    print("Embedding:", embedding)

    # 5 Compare to database
    db_embeddings = np.load('db_embeddings.npy') # shape [N, embed_dim]
    db_labels = np.load('db_labels.npy') # shape [N]

     #compute cosine similarity
    embedding_norm = embedding / np.linalg.norm(embedding)
    db_norm = db_embeddings / np.linalg.norm(db_embeddings, axis=1, keepdims=True)
    similarities = np.dot(db_norm, embedding_norm)

    #Find best match
    best_idx = np.argmax(similarities)

    threshold = 0.7
    if similarities[best_idx] >= threshold:
        print(f"Best match: {db_labels[best_idx]} (similarity: {similarities[best_idx]: .4f})")
    else:
        print(f"No matching. Highest similarity: {similarities[best_idx]:.4f}")

if __name__ == "__main__":
    main()