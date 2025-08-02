import torch
import yaml
from PIL import Image
import argparse
from timm.models import create_model
import numpy as np
from facenet_pytorch import MTCNN
import io
from typing import List, Tuple, Optional
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import models.spikingresformer
import sqlite3

def load_config(config_path):
    "Load YAML config file"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
def load_model(cfg_path: str, device: torch.device = torch.device('cpu')) -> torch.nn.Module:
    "Load the SNNs face recognition model and preprocessing pipeline"
    cfg = load_config(cfg_path)
    input_size = cfg.get('input_size', [3, 112, 112])
    model_name = cfg.get('model', 'spikingresformer_ti')
    checkpoint_path = cfg.get('checkpoint', './logs/checkpoint_best.pth')
    T = cfg.get('T', 4)
    embed_dim = cfg.get('embeded_dim', 512)

    model = create_model(
        model_name,
        T,
        num_classes = embed_dim,
        img_size = input_size[-1]
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()

    return model

def get_embedding(
        model: torch.nn.Module,
        img_bytes: bytes,
        device: torch.device,
) -> np.array:
    
    "Detect face, preprocess, and compute the embedding"
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    # Detect faces
    aligned = detector(img)
    if aligned is None:
        raise ValueError("No face detected")
    
    tensor = aligned.unsqueeze(0).to(device)
    # Infer
    with torch.no_grad():
        emb = model(tensor)
        # Handle temporal outputs
        if emb.dim() == 3: # check for temporal, yes then calculate mean over time
            emb = emb.mean(0, keepdim=True)
    emb = emb.squeeze(0).cpu().numpy() #Changes shape from [1, D] into [D]
    # Normalize so their dot product becomes exactly the cosine of the angle between them.
    emb = emb / np.linalg.norm(emb)
    return emb

def compare_embedding (
        emb: np.array,
        db_embeddings: np.array,
        db_labels: np.array,
        threshold: float
) -> Tuple[Optional[str], float]:
    """
    Compare embedding to database and return best match
    Each user has 3 embeddings (frontal, left, right)
    return the user with best match similarity
    """
    if db_embeddings.size == 0:
        return None, 0.0
    
    # Group every 3 embeddings together (frontal, left, right)
    num_users = len(db_labels)
    assert db_embeddings.shape[0] == num_users * 3

    best_user = None
    best_sim = 0.0

    input_emb = np.array(emb).squeeze()

    for i in range(num_users):
        user_id = db_labels[i]
        pose_embs = db_embeddings[i*3 : (i+1)*3]

        sims = pose_embs.dot(input_emb)
        user_best_sim = float(np.max(sims))

        if user_best_sim >= best_sim:
            best_sim = user_best_sim
            best_user = user_id

    if best_sim >= threshold:
        return best_user, best_sim 
    else:
        return None, best_sim
    
# -------------------------------Fast API App ----------------------------------------------------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    # CORS = Cross - Origin Resource Sharing 
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Globals
model = None
preprocess_fn = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
detector = MTCNN(
    image_size = 112, 
    margin=0, 
    post_process=False, 
    device=device
)
db_embeddings = np.zeros((0, 512), dtype=np.float32)
db_labels: List[str]= []
THRESHOLD = 0.75
DB_PATH = "faces.db"
conn: sqlite3.Connection = None
cursor: sqlite3.Cursor = None
# what are those ?

#Pydantic response model
default_response = {
    "recognized": False,
    "user_id": None,
    "similarity": 0.0,
    "votes": 0
}

# provide json data will be given back
class RecognizeResponse(BaseModel):
    recognized: bool
    user_id: Optional[str]
    similarity: float
    votes: int

@app.on_event("startup")
def startup_event(): # are the positions matter ?
    global model, preprocess_fn, detector, conn, cursor, db_embeddings, db_labels

    # 1 load model, preprocess and face detector
    model = load_model("configs/inference.yaml", device)

    # warm up spiking
    from PIL import Image
    dummy_img = Image.new("RGB", (112, 112), (128, 128, 128)) # last one create shade of grey
    _ = detector(dummy_img)

    # 2 open sqlite and create table
    conn = sqlite3.connect(DB_PATH, check_same_thread=False) #Fastapi might handles requests across threads
    # because requests can come at the same time, in that moment they need to access db so we let check_same_thread False
    cursor = conn.cursor() # use this to execute command in SQL
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            user_id TEXT PRIMARY KEY,
            frontal_emb BLOB NOT NULL,
            left_emb BLOB NOT NULL,
            right_emb BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
    conn.commit() # like GIT, to save everything in db

    # 3 Load existing faces into memory
    cursor.execute("SELECT user_id, frontal_emb, left_emb, right_emb FROM faces")
    rows = cursor.fetchall() #taje all data put into rows var
    db_labels = [row[0] for row in rows]
    # convert each BLOB back into numpy array
    db_embeddings = np.vstack([
        np.frombuffer(emb, dtype=np.float32) # turn bytes back to numpy 
        # loop through all the user in rows and get all the embedding
        for row in rows
        for emb in row[1:4]
    ]) if rows else np.zeros((0, 512), dtype=np.float32)

# register a new face
@app.post("/register")
async def register_face(
    user_id: str =Form(...), # (...) stands for required
    frontal: UploadFile = File(...),
    left: UploadFile = File(...),
    right: UploadFile = File(...),
):
    try:
        files = {
            "frontal": frontal,
            "left": left,
            "right": right
        }
        #Step 1 Read and extracts embedding for 3 poses
        embeddings = {}
        for pose, file in files.items():
            img_bytes = await file.read()

            # Get embedding
            emb = get_embedding(model, img_bytes, device)
            embeddings[pose] = emb.astype(np.float32).tobytes()

        #insert into DB
        cursor.execute(
            """
            INSERT INTO faces (user_id, frontal_emb, left_emb, right_emb) 
            VALUES (?, ?, ?, ?)            
            """,
            (user_id, embeddings["frontal"], embeddings["left"], embeddings["right"]),
        )
        conn.commit()

        global db_embeddings

        cursor.execute("SELECT user_id, frontal_emb, left_emb, right_emb FROM faces")
        rows = cursor.fetchall()
        db_labels.clear()
        db_labels.extend([row[0] for row in rows])
        db_embeddings = np.vstack([
            np.frombuffer(row[i], dtype=np.float32)
            for row in rows
            for i in range(1, 4)
        ])

        return JSONResponse(
            content={
                "status": "success",
                "user_id": user_id,
                "registered_poses": list(files.keys())
            }
        )
    # any exception will run this code block
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")
    
@app.post("/compare", response_model=RecognizeResponse)
async def compare_face(
    images: List[UploadFile] = File(..., description="Send 5 face images")
):
    if len(images) < 5:
        raise HTTPException(400, "Please send exactly 5 images")
    

    #  1 run each image throguh get embed + comparison
    votes = {} #user_id -> count of passes
    sims = [] #collect (user_id, sim) tuples for reporting

    for image in images:
        img_bytes = await image.read()
        try:
            emb = get_embedding(model, img_bytes, device)
        except ValueError: 
            # no face in this one- count as a miss
            continue
        user_id, similarity = compare_embedding(emb, db_embeddings, db_labels, THRESHOLD)
        sims.append((user_id, similarity))
        if user_id is not None:
            votes[user_id] = votes.get(user_id, 0) + 1 # 0 are default params
        
    # 2 decide winner
    best_user, best_votes = None, 0
    for user_id, count in votes.items():
        if count > best_votes:
            best_user, best_votes = user_id, count

    # 3 If someone got 3 mores => recognized
    recognized = (best_votes >=3 )
    best_sim = 0.0
    if recognized:
        best_sim = max(sim for (uid, sim) in sims if uid == best_user)

    return RecognizeResponse(
        recognized=recognized,
        user_id=best_user if recognized else None,
        similarity=best_sim,
        votes = best_votes
    )

@app.get("/healthz")
def healthz():
    return {"status": "ok"}


if __name__ == "__main__":
    "Reads PORT from env (default 8000) so you can overide if needed"
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend:app", host="0.0.0.0", port=port, reload=True)
