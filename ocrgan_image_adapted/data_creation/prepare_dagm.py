import os
import shutil

def prepare_dagm_classes(src_root="data/unprocessed/DAGM", dest_root="data/processed/dagm_processed"):
    for i in range(1, 11):
        class_name = f"Class{i}"
        dagm_name = f"dagm_{i}"
        print("Processing class:", class_name)
        class_dir = os.path.join(src_root, class_name)
        train_src = os.path.join(class_dir, "Train")
        train_label = os.path.join(train_src, "Label")
        test_src = os.path.join(class_dir, "Test")
        test_label = os.path.join(test_src, "Label")

        # Prepare output folders
        train_good = os.path.join(dest_root, dagm_name, "train", "good")
        test_good = os.path.join(dest_root, dagm_name, "test", "good")
        test_bad = os.path.join(dest_root, dagm_name, "test", "bad")
        os.makedirs(train_good, exist_ok=True)
        os.makedirs(test_good, exist_ok=True)
        os.makedirs(test_bad, exist_ok=True)

        # --- TRAIN ---
        label_files = set()
        if os.path.exists(train_label):
            for f in os.listdir(train_label):
                if f.lower().endswith('_label.png'):
                    label_files.add(os.path.splitext(f)[0].replace('_label', ''))

        if os.path.exists(train_src):
            for img in os.listdir(train_src):
                img_path = os.path.join(train_src, img)
                if img == "Label" or not img.lower().endswith('.png'):
                    continue
                img_base = os.path.splitext(img)[0]
                if img_base not in label_files:
                    shutil.copy2(img_path, os.path.join(train_good, img))

        # --- TEST ---
        test_label_files = set()
        if os.path.exists(test_label):
            for f in os.listdir(test_label):
                if f.lower().endswith('_label.png'):
                    test_label_files.add(os.path.splitext(f)[0].replace('_label', ''))

        if os.path.exists(test_src):
            for img in os.listdir(test_src):
                img_path = os.path.join(test_src, img)
                if img == "Label" or not img.lower().endswith('.png'):
                    continue
                img_base = os.path.splitext(img)[0]
                if img_base in test_label_files:
                    shutil.copy2(img_path, os.path.join(test_bad, img))
                else:
                    shutil.copy2(img_path, os.path.join(test_good, img))

if __name__ == "__main__":
    print("Processing the DAGM classes to be ready for OCRGAN :")
    prepare_dagm_classes()
    print("Processing done.")