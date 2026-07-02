"""Quick sanity check with a synthetic dataset before real training."""
import os
from PIL import Image
import numpy as np

def make_fake_dataset(root="fake_data", n_classes=3, n_per_class=20, size=64):
    os.makedirs(root, exist_ok=True)
    for c in range(n_classes):
        class_dir = os.path.join(root, f"class_{c}")
        os.makedirs(class_dir, exist_ok=True)
        for i in range(n_per_class):
            arr = (np.random.rand(size, size, 3) * 255).astype("uint8")
            Image.fromarray(arr).save(os.path.join(class_dir, f"img_{i}.jpg"))

if __name__ == "__main__":
    make_fake_dataset()
    print("Fake dataset created at ./fake_data")
