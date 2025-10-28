import os
import requests
import json
from tqdm import tqdm

BASE_URL = "https://api.isic-archive.com/api/v2"
DATA_DIR = os.path.join("datasets", "ISIC2024")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
META_FILE = os.path.join(DATA_DIR, "metadata.json")

os.makedirs(IMAGES_DIR, exist_ok=True)

def fetch_metadata(limit=200):

    print("[INFO] Fetching ISIC 2024 metadata...")
    url = f"{BASE_URL}/images?limit={limit}"
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Failed to fetch metadata: {response.text}")

    data = response.json()
    with open(META_FILE, "w") as f:
        json.dump(data, f, indent=4)
    print(f"[INFO] Saved metadata for {len(data.get('results', []))} images to {META_FILE}")
    return data.get("results", [])


def download_images(image_metadata):

    print(f"[INFO] Downloading {len(image_metadata)} ISIC 2024 images...")
    for item in tqdm(image_metadata):
        image_id = item["isic_id"]
        image_url = item["files"]["full"]["url"]
        save_path = os.path.join(IMAGES_DIR, f"{image_id}.jpg")

        if os.path.exists(save_path):
            continue

        img_data = requests.get(image_url)
        if img_data.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(img_data.content)
        else:
            print(f"[WARN] Failed to download {image_id}")

    print(f"[INFO] All images saved to {IMAGES_DIR}")


if __name__ == "__main__":
    metadata = fetch_metadata(limit=300)
    download_images(metadata)
