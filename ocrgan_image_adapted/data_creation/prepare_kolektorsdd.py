import os
import shutil
from PIL import Image
import numpy as np
import random

def is_mask_anomalous(label_path):
    """Return True if the label (bmp) contains any white pixel (anomaly)."""
    with Image.open(label_path) as img:
        arr = np.array(img)
        return np.any(arr > 0)

def prepare_kolektor_sdd(src_root, dest_root, seed=42):
    random.seed(seed)
    kos_dirs = [d for d in os.listdir(src_root) if d.startswith('kos')]
    os.makedirs(dest_root, exist_ok=True)

    for kos in kos_dirs:
        print(f"Processing {kos}...")

        src_dir = os.path.join(src_root, kos)
        good_imgs = []
        bad_imgs = []

        # Sort all images into good/bad
        for fname in os.listdir(src_dir):
            if fname.endswith('.jpg') and not fname.endswith('_label.jpg'):
                base = fname[:-4]
                label_path = os.path.join(src_dir, f"{base}_label.bmp")
                img_path = os.path.join(src_dir, fname)
                if not os.path.exists(label_path):
                    print(f"  Warning: label not found for {img_path}, skipping.")
                    continue
                if is_mask_anomalous(label_path):
                    bad_imgs.append((img_path, fname))
                else:
                    good_imgs.append((img_path, fname))

        print(f"  Found {len(good_imgs)} good and {len(bad_imgs)} bad images.")

        # Prepare output dirs
        train_good_dir = os.path.join(dest_root, kos, "train", "good")
        test_good_dir = os.path.join(dest_root, kos, "test", "good")
        test_bad_dir = os.path.join(dest_root, kos, "test", "bad")
        for d in [train_good_dir, test_good_dir, test_bad_dir]:
            os.makedirs(d, exist_ok=True)

        # Test set: all bads + same number of goods
        num_test = len(bad_imgs)
        if num_test > len(good_imgs):
            print(f"  Not enough good samples to balance test! Only {len(good_imgs)} available, using all.")
            test_good_samples = good_imgs
        else:
            test_good_samples = random.sample(good_imgs, num_test)

        # Remove test goods from the train goods
        test_good_set = set([x[1] for x in test_good_samples])
        train_good_samples = [x for x in good_imgs if x[1] not in test_good_set]

        # Copy images
        for img_path, fname in bad_imgs:
            shutil.copy2(img_path, os.path.join(test_bad_dir, fname))
        for img_path, fname in test_good_samples:
            shutil.copy2(img_path, os.path.join(test_good_dir, fname))
        for img_path, fname in train_good_samples:
            shutil.copy2(img_path, os.path.join(train_good_dir, fname))

        print(f"  Copied {len(train_good_samples)} to train/good, {len(test_good_samples)} to test/good, {len(bad_imgs)} to test/bad.")

if __name__ == "__main__":
    src_root = "data/unprocessed/KolektorSDD"
    dest_root = "data/processed/KolektorSDD_processed"
    prepare_kolektor_sdd(src_root, dest_root)
    print("KolektorSDD dataset prepared.")