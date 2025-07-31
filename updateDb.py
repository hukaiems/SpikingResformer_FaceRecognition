import os
import torch
from inference import preprocess_image, load_config, save_to_database
from timm.models import create_model
import numpy as np

def batch_add_images_to_db(folder_path, db_embeddings_path="db_embeddings.npy", db_labels_path='db_labels.npy', config_path='configs/inference.yaml'):
    cfg = load_config(config_path)
    input_size = cfg.get('input_size', [3, 112, 112])
    model_name = cfg.get('model', 'spikingresformer_ti')
    checkpoint_path = cfg.get('checkpoint', './logs/checkpoint_best.pth')
    T = cfg.get('T', 4)

    model = create_model(model_name, T=T, num_classes=cfg.get('embed_dim', 512), img_size=input_size[-1])
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img_tensor = preprocess_image(img_path, input_size).cuda()
        with torch.no_grad():
            embedding = model(img_tensor)
            if embedding.dim() == 3:
                embedding = embedding.mean(0)
            embedding = embedding.squeeze(0).cpu().numpy()
        label = os.path.splitext(img_file)[0]  # Use filename (without extension) as label
        save_to_database(embedding, label, db_embeddings_path, db_labels_path)
        print(f"Added {img_file} as '{label}' to database.")