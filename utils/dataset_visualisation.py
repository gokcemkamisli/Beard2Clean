import os
import matplotlib.pyplot as plt
from PIL import Image
import random 
DATASET_PATH = "dataset"

def preview_image_pairs(num_pairs: int = 10):
    all_pairs = []
    for filename in os.listdir(DATASET_PATH):
        if "_clean.png" in filename:
            pair_id = filename.split("_clean.png")[0]
            beard_image_path = os.path.join(DATASET_PATH, f"{pair_id}_beard.png")
            clean_image_path = os.path.join(DATASET_PATH, filename)

            if os.path.exists(beard_image_path):  
                all_pairs.append((clean_image_path, beard_image_path))

    random.shuffle(all_pairs)
    selected_pairs = all_pairs[:num_pairs]

    plt.figure(figsize=(10, num_pairs * 5))
    
    for i, (clean_path, beard_path) in enumerate(selected_pairs):
        clean_image = Image.open(clean_path)
        beard_image = Image.open(beard_path)

        plt.subplot(num_pairs, 2, i * 2 + 1)
        plt.imshow(clean_image)
        plt.axis('off')
        plt.title(f"Pair {i+1} - Clean")

        plt.subplot(num_pairs, 2, i * 2 + 2)
        plt.imshow(beard_image)
        plt.axis('off')
        plt.title(f"Pair {i+1} - Beard")

    plt.tight_layout()
    plt.show()