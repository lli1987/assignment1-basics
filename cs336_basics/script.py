import os
import random
import shutil


def copy_random_subset(
    src_dir: str,
    dst_dir: str,
    fraction: float = 0.1,
    seed: int = 42,
):
    random.seed(seed)

    os.makedirs(dst_dir, exist_ok=True)

    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

    num_to_copy = int(len(files) * fraction)
    selected_files = random.sample(files, num_to_copy)

    print(f"Total files: {len(files)}")
    print(f"Copying {num_to_copy} files ({fraction * 100:.1f}%)")

    for fname in selected_files:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        shutil.copy2(src_path, dst_path)

    print("Done.")


if __name__ == "__main__":
    SRC_DIR = "/Users/luyaoli/code/cs336/assignment1-basics/owt_text"
    DST_DIR = "/Users/luyaoli/code/cs336/assignment1-basics/owt_text_1pct"
    copy_random_subset(SRC_DIR, DST_DIR, fraction=0.01)
