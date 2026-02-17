import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tqdm import tqdm 
import torch
from transformers import AutoImageProcessor, AutoModel

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def get_embeddings_batch_ResNet50(df, batch_size=64):
    """
    Computes ResNet50 embeddings for all chips in the dataframe with a progress bar.
    """
    
    print("--- Initializing ResNet50 ---")
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    # Initialize the feature column
    df['img_feature'] = None
    df['img_feature'] = df['img_feature'].astype(object)
    
    total_samples = len(df)

    pbar = tqdm(total=total_samples, desc="Generating Embeddings", unit="img")

    for i in range(0, total_samples, batch_size):
        batch_df = df.iloc[i:i+batch_size]
        imgs, indices = [], []
        
        for idx, row in batch_df.iterrows():
            try:
                img = np.load(row['chip_path'])
                if img.shape[-1] > 3: 
                    img = img[..., :3]
                
                img = preprocess_input(img.astype('float32'))
                imgs.append(img)
                indices.append(idx)
            except Exception as e:
                continue
            
        if imgs:
            # Predict
            preds = model.predict(np.array(imgs), verbose=0)
            
            # Storage
            for j, real_idx in enumerate(indices):
                df.at[real_idx, 'img_feature'] = preds[j]
        
   
        pbar.update(len(imgs))
        
    pbar.close() 
    return df.dropna(subset=['img_feature'])



def get_embeddings_batch_SatDINO(df, batch_size=32):
    """
    Computes SatDINO (ViT-base) embeddings for all chips in the dataframe.
    Model: https://huggingface.co/strakajk/satdino-vit_base-16
    """
    print("--- Initializing SatDINO (ViT-Base-16) ---")
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "strakajk/satdino-vit_base-16"
    
    # Load Processor and Model
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval() # Set to evaluation mode

    df['img_feature'] = None
    df['img_feature'] = df['img_feature'].astype(object)
    
    total_samples = len(df)
    pbar = tqdm(total=total_samples, desc="Generating SatDINO Embeddings", unit="img")

    for i in range(0, total_samples, batch_size):
        batch_df = df.iloc[i:i+batch_size]
        imgs, indices = [], []
        
        for idx, row in batch_df.iterrows():
            try:
                # Load .npy chip
                img = np.load(row['chip_path'])
                if img.shape[-1] > 3: 
                    img = img[..., :3] # Take RGB only
                
                # SatDINO expects 0-255 range usually, or the processor handles it
                imgs.append(img)
                indices.append(idx)
            except Exception:
                continue
            
        if imgs:
            # Preprocessing
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            
            with torch.no_grad():
                # Extract the [CLS] token or pool the hidden states
                outputs = model(**inputs)
                # SatDINO returns (batch_size, sequence_length, hidden_size)
                # We take the first token [CLS] which represents the global image features
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Storage
            for j, real_idx in enumerate(indices):
                df.at[real_idx, 'img_feature'] = embeddings[j]
        
        pbar.update(len(imgs))
        
    pbar.close()
    return df.dropna(subset=['img_feature'])

if __name__ == "__main__":
    CACHE_FILE = "../data/extracted/metadata_cache.csv"
    OUTPUT_FILE = "../data/extracted/embeddings.pkl"
    
    if os.path.exists(CACHE_FILE):
        df = pd.read_csv(CACHE_FILE)
        print(f"Loaded {len(df)} entries from cache.")
        
        df_emb = get_embeddings_batch_SatDINO(df)
        
        # Save results
        df_emb.to_pickle(OUTPUT_FILE)
        print(f"Successfully saved embeddings to {OUTPUT_FILE}")
    else:
        print(f"Error: {CACHE_FILE} not found.")