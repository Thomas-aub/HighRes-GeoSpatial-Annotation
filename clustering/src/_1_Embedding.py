import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tqdm import tqdm 


def get_embeddings_batch(df, batch_size=64):
    """
    Computes ResNet50 embeddings for all chips in the dataframe with a progress bar.
    """
    print("--- Initializing ResNet50 ---")
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    # Initialize the feature column
    df['img_feature'] = None
    df['img_feature'] = df['img_feature'].astype(object)
    
    total_samples = len(df)
    
    # Configuration de tqdm :
    # total : nombre total d'itérations
    # desc : texte affiché devant la barre
    # unit : l'unité de mesure (ici des lots d'images)
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
        
        # Mise à jour de la barre de progression par la taille du batch réel
        pbar.update(len(imgs))
        
    pbar.close() # Fermeture de la barre à la fin
    return df.dropna(subset=['img_feature'])

if __name__ == "__main__":
    # Correction des chemins relatifs selon votre structure de dossiers
    CACHE_FILE = "../data/extracted/metadata_cache.csv"
    OUTPUT_FILE = "../data/extracted/embeddings.pkl"
    
    if os.path.exists(CACHE_FILE):
        df = pd.read_csv(CACHE_FILE)
        print(f"Loaded {len(df)} entries from cache.")
        
        df_emb = get_embeddings_batch(df)
        
        # Save results
        df_emb.to_pickle(OUTPUT_FILE)
        print(f"Successfully saved embeddings to {OUTPUT_FILE}")
    else:
        print(f"Error: {CACHE_FILE} not found.")