import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from transformers import AutoModel
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tqdm import tqdm
from PIL import Image
from typing import Literal

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_embeddings_batch(
    df: pd.DataFrame, 
    model_name: Literal["SatDINO", "ResNet50"] = "SatDINO",
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Computes embeddings for all tiles in the dataframe using the specified model.
    """
    if model_name == "SatDINO":
        return get_embeddings_batch_SatDINO(df, batch_size)
    elif model_name == "ResNet50":
        return get_embeddings_batch_ResNet50(df, batch_size)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

def get_embeddings_batch_SatDINO(df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    """
    Computes embeddings using the SatDINO ViT-Base-16 model on standard tiles.
    """
    print("--- Initializing SatDINO (ViT-Base-16) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "strakajk/satdino-vit_base-16"

    # Temporary patch for a potential issue with torch.linspace in some environments
    _orig_linspace = torch.linspace
    def _safe_linspace(*args, **kwargs):
        kwargs["device"] = "cpu"
        return _orig_linspace(*args, **kwargs)
    torch.linspace = _safe_linspace

    try:
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
            device_map=None,
        ).to(device)
    finally:
        torch.linspace = _orig_linspace  # Restore original function

    model.eval()

    # Preprocessing pipeline for SatDINO
    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    df['img_feature'] = None
    df['img_feature'] = df['img_feature'].astype(object)
    
    total_samples = len(df)
    pbar = tqdm(total=total_samples, desc="Generating SatDINO Embeddings", unit="img")

    for i in range(0, total_samples, batch_size):
        batch_df = df.iloc[i:i+batch_size]
        batch_tensors = []
        indices = []
        
        for idx, row in batch_df.iterrows():
            try:
                # Load the tile directly
                img = Image.open(row['tile_path']).convert('RGB')
                batch_tensors.append(preprocess(img))
                indices.append(idx)
            except Exception as e:
                print(f"ERROR [{idx}] {row['tile_path']}: {type(e).__name__}: {e}")
                continue
            
        if batch_tensors:
            input_batch = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                outputs = model(input_batch)
                
                if hasattr(outputs, 'last_hidden_state'):
                    hidden = outputs.last_hidden_state
                    embeddings = (hidden[:, 0, :] if hidden.dim() == 3 else hidden).cpu().numpy()
                else:
                    embeddings = (outputs[:, 0, :] if outputs.dim() == 3 else outputs).cpu().numpy()

            for j, real_idx in enumerate(indices):
                df.at[real_idx, 'img_feature'] = embeddings[j]
        
        pbar.update(len(batch_df))
        
    pbar.close()
    return df.dropna(subset=['img_feature'])


def get_embeddings_batch_ResNet50(df: pd.DataFrame, batch_size: int = 64) -> pd.DataFrame:
    """
    Computes embeddings using the ResNet50 model from Keras on standard tiles.
    """
    print("--- Initializing ResNet50 ---")
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    df['img_feature'] = None
    df['img_feature'] = df['img_feature'].astype(object)
    
    total_samples = len(df)
    pbar = tqdm(total=total_samples, desc="Generating ResNet50 Embeddings", unit="img")

    for i in range(0, total_samples, batch_size):
        batch_df = df.iloc[i:i+batch_size]
        imgs, indices = [], []
        
        for idx, row in batch_df.iterrows():
            try:
                img = Image.open(row['tile_path']).convert('RGB')
                # Resize to standard ResNet input size
                img = img.resize((224, 224))
                
                # Convert to numpy array and preprocess for ResNet
                img_array = np.array(img, dtype=np.float32)
                img_array = preprocess_input(img_array)
                
                imgs.append(img_array)
                indices.append(idx)
            except Exception as e:
                print(f"ERROR [{idx}] {row['tile_path']}: {type(e).__name__}: {e}")
                continue
            
        if imgs:
            preds = model.predict(np.array(imgs), verbose=0)
            
            for j, real_idx in enumerate(indices):
                df.at[real_idx, 'img_feature'] = preds[j]
        
        pbar.update(len(batch_df))
        
    pbar.close() 
    return df.dropna(subset=['img_feature'])


if __name__ == "__main__":
    METADATA_FILE = "./metadata.csv"
    OUTPUT_FILE = "./data/embeddings.pkl"
    TILES_DIR = "./tiles/"
    
    os.makedirs("./data", exist_ok=True)

    if os.path.exists(METADATA_FILE):
        df = pd.read_csv(METADATA_FILE)
        print(f"Loaded {len(df)} entries from {METADATA_FILE}.")

        # Create a full path to the tile for the embedding function
        df['tile_path'] = df['filename'].apply(lambda x: os.path.join(TILES_DIR, x))
        
        # Filter out rows where the image doesn't actually exist on disk
        df = df[df['tile_path'].apply(os.path.exists)]
        print(f"Found {len(df)} valid tiles on disk.")
        
        # --- CHOOSE YOUR MODEL HERE ---
        # Options: "SatDINO" or "ResNet50"
        CHOSEN_MODEL = "SatDINO"
        
        df_emb = get_embeddings_batch(df, model_name=CHOSEN_MODEL)
        
        # Save results
        df_emb.to_pickle(OUTPUT_FILE)
        print(f"Successfully saved embeddings to {OUTPUT_FILE}")
    else:
        print(f"Error: {METADATA_FILE} not found.")