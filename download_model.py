"""
Downloads the embedding model from HuggingFace and saves it to models/gte-modernbert-base-fp16.
Run this once before using the pipeline: python3 download_model.py
"""

import os
import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "Alibaba-NLP/gte-modernbert-base"
SAVE_PATH = "models/gte-modernbert-base-fp16"


def main():
    if os.path.exists(os.path.join(SAVE_PATH, "model.safetensors")):
        print(f"Model already exists at {SAVE_PATH}, skipping download.")
        return

    print(f"Downloading {MODEL_ID} from HuggingFace...")
    model = SentenceTransformer(MODEL_ID, model_kwargs={"torch_dtype": torch.float16})
    os.makedirs(SAVE_PATH, exist_ok=True)
    model.save(SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
