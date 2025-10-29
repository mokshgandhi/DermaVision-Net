import os
import shutil
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

HAM_PATH = os.path.join("datasets", "HAM10000")
HAM_CSV = os.path.join(HAM_PATH, "HAM10000_metadata.csv")
HAM_IMG_DIRS = [
    os.path.join(HAM_PATH, "HAM10000_images_part_1"),
    os.path.join(HAM_PATH, "HAM10000_images_part_2")
]

def organize_ham10000():
    print("[INFO] Checking HAM10000 dataset structure...")
    example_class_dir = os.path.join(HAM_PATH, "mel")
    if os.path.exists(example_class_dir):
        print("[INFO] HAM10000 already organized into class folders.")
        return

    if not os.path.exists(HAM_CSV):
        raise FileNotFoundError(f"[ERROR] Metadata file not found: {HAM_CSV}")

    print("[INFO] Organizing HAM10000 images into class folders...")
    df = pd.read_csv(HAM_CSV)
    total = len(df)

    for i, row in df.iterrows():
        image_id = row["image_id"]
        label = row["dx"]
        src = None

        for d in HAM_IMG_DIRS:
            img_path = os.path.join(d, f"{image_id}.jpg")
            if os.path.exists(img_path):
                src = img_path
                break

        if src:
            dst_dir = os.path.join(HAM_PATH, label)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(src, os.path.join(dst_dir, f"{image_id}.jpg"))
        else:
            print(f"[WARN] Missing image for ID: {image_id}")

        if (i + 1) % 1000 == 0:
            print(f"[INFO] Processed {i+1}/{total} images...")

    print("[INFO] HAM10000 reorganization complete ✅")

def create_data_generators(data_dir, img_size=(224, 224), batch_size=32):
    datagen_train = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True
    )

    train_gen = datagen_train.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    val_gen = datagen_train.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    return train_gen, val_gen


if __name__ == "__main__":
    organize_ham10000()
    print("[INFO] Preparing HAM10000 data generators...")
    ham_train, ham_val = create_data_generators(HAM_PATH)
    print(f"[INFO] Training batches: {len(ham_train)}, Validation batches: {len(ham_val)}")
