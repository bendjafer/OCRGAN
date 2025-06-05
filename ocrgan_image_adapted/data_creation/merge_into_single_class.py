import os
import shutil
from glob import glob

def merge_datasets(classes, src_root, dest_root):
    subsets = [
        ("train", "good"),
        ("test", "good"),
        ("test", "bad")
    ]
    # Create destination folders
    for subset, status in subsets:
        out_dir = os.path.join(dest_root, subset, status)
        os.makedirs(out_dir, exist_ok=True)

    for class_name in classes:
        print(f"Processing {class_name}...")
        for subset, status in subsets:
            src_dir = os.path.join(src_root, class_name, subset, status)
            if not os.path.exists(src_dir):
                continue
            # Match both .png and .PNG files
            img_paths = sorted(glob(os.path.join(src_dir, "*.png"))) + sorted(glob(os.path.join(src_dir, "*.PNG"))) + sorted(glob(os.path.join(src_dir, "*.jpg"))) + sorted(glob(os.path.join(src_dir, "*.JPG")))
            for img_path in img_paths:
                img_name = os.path.basename(img_path)
                new_img_name = f"{class_name}_{img_name}"
                dest_dir = os.path.join(dest_root, subset, status)
                dest_path = os.path.join(dest_dir, new_img_name)
                shutil.copy2(img_path, dest_path)
                print(f"  Copied {img_path} -> {dest_path}")

if __name__ == "__main__":
    # === MVTec ===
    mvtec_classes = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper"
    ]
    mvtec_src_root = "data/processed/mvtec_processed"
    mvtec_dest_root = "data/merged/mvtec_merged"

    
    print("Merging the MVTEC classes:")
    merge_datasets(mvtec_classes, mvtec_src_root, mvtec_dest_root)
    print("MVTEC classes merged.\n")

    # === DAGM ===
    dagm_classes = [f"dagm_{i}" for i in range(1, 11)]
    dagm_src_root = "data/processed/dagm_processed"
    dagm_dest_root = "data/merged/dagm_merged"
    print("Merging the DAGM classes:")
    merge_datasets(dagm_classes, dagm_src_root, dagm_dest_root)
    print("DAGM classes merged.\n")

    # === KolektorSDD ===
    kolektor_classes = [f"kos{str(i).zfill(2)}" for i in range(1, 51)]  # kos01, kos02, ..., kos50
    kolektor_src_root = "data/processed/kolektorsdd_processed"
    kolektor_dest_root = "data/merged/kolektorsdd_merged"
    print("Merging the KolektorSDD classes:")
    merge_datasets(kolektor_classes, kolektor_src_root, kolektor_dest_root)
    print("KolektorSDD classes merged.\n")

