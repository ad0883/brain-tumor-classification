import os
import cv2
import numpy as np

DATA_DIR = "data/raw/archive"
OUTPUT_DIR = "data/processed"
IMG_SIZE = 224

splits = ["Training", "Testing"]
classes = ["glioma", "meningioma", "pituitary", "notumor"]

label_map = {
    "glioma": 0,
    "meningioma": 1,
    "pituitary": 2,
    "notumor": 3
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess(split):
    X, y = [], []

    for label in classes:
        folder = os.path.join(DATA_DIR, split, label)

        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            X.append(img)
            y.append(label_map[label])

    return np.array(X), np.array(y)

print("Preprocessing training data...")
X_train, y_train = preprocess("Training")

print("Preprocessing testing data...")
X_test, y_test = preprocess("Testing")

np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

print("Preprocessing complete!")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
