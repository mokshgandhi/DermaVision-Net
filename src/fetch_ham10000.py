import os
import zipfile
import subprocess
from tqdm import tqdm

DATA_DIR = os.path.join("datasets", "HAM10000")
ZIP_PATH = os.path.join(DATA_DIR, "ham10000.zip")

def download_ham10000():
    
    print("[INFO] Downloading HAM10000 dataset from Kaggle...")

    os.makedirs(DATA_DIR, exist_ok=True)
    
    command = [
        "kaggle", "datasets", "download",
        "-d", "kmader/skin-cancer-mnist-ham10000",
        "-p", DATA_DIR
    ]

    subprocess.run(command, check=True)
    print("[INFO] Download completed.")


def extract_zip():
    
    zips = [f for f in os.listdir(DATA_DIR) if f.endswith(".zip")]
    if not zips:
        print("[WARN] No ZIP file found in datasets/HAM10000/")
        return
    
    for zip_file in tqdm(zips, desc="[INFO] Extracting files"):
        zip_path = os.path.join(DATA_DIR, zip_file)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print(f"[INFO] Extracted {zip_file}")
        os.remove(zip_path)  


if __name__ == "__main__":
    download_ham10000()
    extract_zip()
    print(f"[SUCCESS] HAM10000 dataset ready in {DATA_DIR}/")
