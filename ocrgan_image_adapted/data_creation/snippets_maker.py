import os
import shutil
import numpy as np
from PIL import Image
import re

SNIPPET_LENGTH = 16  # You may adjust this as needed

def get_sorted_frames(folder, ext):
    """Return sorted list of frame filenames based on numeric content."""
    files = [f for f in os.listdir(folder) if f.endswith(ext)]
    def extract_number(filename):
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return int(numbers[-1])
        return 0
    return sorted(files, key=extract_number)

def snippet_chunks(frame_list, k):
    """Generate consecutive frame chunks of length k."""
    for i in range(0, len(frame_list) - k + 1):
        yield frame_list[i:i+k], i

def snippet_has_anomaly(mask_paths):
    """Check if any mask in the snippet contains white pixels (anomaly)."""
    for path in mask_paths:
        try:
            mask = np.array(Image.open(path).convert("L"))
            if np.any(mask > 0):
                return True
        except Exception as e:
            print(f"Warning: Error processing mask {path}: {e}")
    return False

def copy_snippet(src_folder, frame_names, dest_folder, snippet_name):
    """Copy a set of frames to a snippet folder."""
    snippet_dir = os.path.join(dest_folder, snippet_name)
    os.makedirs(snippet_dir, exist_ok=True)
    for fname in frame_names:
        shutil.copy2(os.path.join(src_folder, fname), os.path.join(snippet_dir, fname))
    return snippet_dir

def process_train(train_dir, out_dir, snippet_length):
    """Process training data into snippets (all good)."""
    out_train_good = os.path.join(out_dir, "train", "good")
    os.makedirs(out_train_good, exist_ok=True)
    snippet_counter = 0

    for seq in sorted(os.listdir(train_dir)):
        seq_path = os.path.join(train_dir, seq)
        if not os.path.isdir(seq_path):
            continue
        frames = get_sorted_frames(seq_path, ".tif")
        for snippet, idx in snippet_chunks(frames, snippet_length):
            snippet_name = f"{seq}_snippet_{idx:04d}"
            copy_snippet(seq_path, snippet, out_train_good, snippet_name)
            snippet_counter += 1
    print(f"Train: Created {snippet_counter} good snippets in {out_train_good}")

def process_test(test_dir, out_dir, snippet_length):
    """Process test data into good/bad snippets based on ground truth."""
    out_good = os.path.join(out_dir, "test", "good")
    out_bad = os.path.join(out_dir, "test", "bad")
    os.makedirs(out_good, exist_ok=True)
    os.makedirs(out_bad, exist_ok=True)
    snippet_counter_good = 0
    snippet_counter_bad = 0

    for seq in sorted(os.listdir(test_dir)):
        # Only process Test sequences, not their ground truth folders
        if not (seq.startswith("Test") and not seq.endswith("_gt")):
            continue

        seq_path = os.path.join(test_dir, seq)
        gt_seq = seq + "_gt"
        gt_path = os.path.join(test_dir, gt_seq)

        if not os.path.isdir(seq_path) or not os.path.isdir(gt_path):
            print(f"Warning: Missing folder for {seq} or {gt_seq}")
            continue

        frames = get_sorted_frames(seq_path, ".tif")
        masks = get_sorted_frames(gt_path, ".bmp")

        if len(frames) != len(masks):
            print(f"Warning: Frame count ({len(frames)}) and mask count ({len(masks)}) mismatch for {seq}")

        for snippet, idx in snippet_chunks(frames, snippet_length):
            # Get corresponding mask paths
            mask_paths = [os.path.join(gt_path, masks[min(i, len(masks)-1)]) for i in range(idx, idx+snippet_length)]
            snippet_name = f"{seq}_snippet_{idx:04d}"

            if snippet_has_anomaly(mask_paths):
                copy_snippet(seq_path, snippet, out_bad, snippet_name)
                snippet_counter_bad += 1
            else:
                copy_snippet(seq_path, snippet, out_good, snippet_name)
                snippet_counter_good += 1

    print(f"Test: Created {snippet_counter_good} good and {snippet_counter_bad} bad snippets.")

if __name__ == "__main__":
    UCSDPED1_ROOT = "../data/UCSDped1"
    OUT_ROOT = "../data/UCSD"

    TRAIN_DIR = os.path.join(UCSDPED1_ROOT, "Train")
    TEST_DIR = os.path.join(UCSDPED1_ROOT, "Test")

    print(f"Processing training data with snippet length {SNIPPET_LENGTH}...")
    process_train(TRAIN_DIR, OUT_ROOT, SNIPPET_LENGTH)

    print(f"Processing test data with snippet length {SNIPPET_LENGTH}...")
    process_test(TEST_DIR, OUT_ROOT, SNIPPET_LENGTH)

    print("Done!")