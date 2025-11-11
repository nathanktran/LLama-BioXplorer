import numpy as np

# Path to your training texts file
train_texts_path = "/p/realai/sneha/cornet2/CorNet/data/Mesh-2022-pubMed/train_texts.npy"

# Load the training texts
train_texts = np.load(train_texts_path)

# Determine the validation size (10% of the training size)
valid_size = int(len(train_texts) * 0.1)

print(f"Validation size: {valid_size}")
