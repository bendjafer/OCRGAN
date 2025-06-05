import os
import shutil
from glob import glob

def prepare_mvtec(src_root, dest_root):
    # List all class folders
    for class_name in os.listdir(src_root):
        class_src = os.path.join(src_root, class_name)
        if not os.path.isdir(class_src):
            continue

        print(f"Processing class: {class_name}")

        class_dest = os.path.join(dest_root, class_name)
        train_src_good = os.path.join(class_src, "train", "good")
        train_dest_good = os.path.join(class_dest, "train", "good")
        test_src = os.path.join(class_src, "test")
        test_dest_good = os.path.join(class_dest, "test", "good")
        test_dest_bad = os.path.join(class_dest, "test", "bad")

        # Remove dest class dir if exists to start clean
        if os.path.exists(class_dest):
            shutil.rmtree(class_dest)

        # Copy train/good as is, with class_name prefix
        if os.path.exists(train_src_good):
            os.makedirs(train_dest_good, exist_ok=True)
            for img_path in sorted(glob(os.path.join(train_src_good, "*.png")) + glob(os.path.join(train_src_good, "*.PNG"))):
                img_name = os.path.basename(img_path)
                new_img_name = f"{class_name}_{img_name}"
                shutil.copy2(img_path, os.path.join(train_dest_good, new_img_name))

        # Prepare test/good and test/bad
        if os.path.exists(test_src):
            # Copy test/good as is, with class_name prefix
            good_src = os.path.join(test_src, "good")
            if os.path.exists(good_src):
                os.makedirs(test_dest_good, exist_ok=True)
                for img_path in sorted(glob(os.path.join(good_src, "*.png")) + glob(os.path.join(good_src, "*.PNG"))):
                    img_name = os.path.basename(img_path)
                    new_img_name = f"{class_name}_{img_name}"
                    shutil.copy2(img_path, os.path.join(test_dest_good, new_img_name))

            # Merge all other test defect folders into test/bad, with class_name and defect_type prefix
            for defect_type in os.listdir(test_src):
                defect_dir = os.path.join(test_src, defect_type)
                if defect_type == "good" or not os.path.isdir(defect_dir):
                    continue
                os.makedirs(test_dest_bad, exist_ok=True)
                for img_path in sorted(glob(os.path.join(defect_dir, "*.png")) + glob(os.path.join(defect_dir, "*.PNG"))):
                    img_name = os.path.basename(img_path)
                    new_img_name = f"{class_name}_{defect_type}_{img_name}"
                    shutil.copy2(img_path, os.path.join(test_dest_bad, new_img_name))

        # Remove ground_truth and txt files in destination
        for root, dirs, files in os.walk(class_dest):
            # Remove ground_truth directories
            for d in dirs:
                if d == "ground_truth":
                    shutil.rmtree(os.path.join(root, d))
            # Remove .txt files
            for f in files:
                if f.lower().endswith(".txt"):
                    os.remove(os.path.join(root, f))

if __name__ == "__main__":
    src_root = "data/unprocessed/mvtec"
    dest_root = "data/processed/mvtec_processed"
    prepare_mvtec(src_root, dest_root)
    print("MVTec dataset prepared.")