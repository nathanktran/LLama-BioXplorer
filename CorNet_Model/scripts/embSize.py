import numpy as np

# Path to your embedding initialization file
emb_init_path = "/p/realai/sneha/cornet2/CorNet/data/Mesh-2022-pubMed/emb_init.npy"

# Load the embeddings
embeddings = np.load(emb_init_path)

# Get the embedding size
emb_size = embeddings.shape[1]

print(f"Embedding size: {emb_size}")
