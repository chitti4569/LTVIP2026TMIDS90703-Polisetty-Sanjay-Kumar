import os
import shutil
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# -----------------------------
# PATHS
# -----------------------------
PROJECT_DIR = r"C:\LT Project"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
TRAIN_IMAGES = os.path.join(DATA_DIR, "train")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")

# -----------------------------
# LOAD LABELS
# -----------------------------
labels = pd.read_csv(LABELS_CSV)

# -----------------------------
# CREATE SUBSET STRUCTURE
# -----------------------------
BASE_DIR = os.path.join(PROJECT_DIR, "subset")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
os.makedirs(TRAIN_DIR, exist_ok=True)

# Create breed folders
for breed in labels['breed'].unique():
    os.makedirs(os.path.join(TRAIN_DIR, breed), exist_ok=True)

# Copy images into folders
for _, row in labels.iterrows():
    src = os.path.join(TRAIN_IMAGES, f"{row['id']}.jpg")
    dst = os.path.join(TRAIN_DIR, row['breed'], f"{row['id']}.jpg")
    if os.path.exists(src):
        shutil.copy(src, dst)

# -----------------------------
# DATA GENERATOR
# -----------------------------
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

NUM_CLASSES = train_generator.num_classes
print("Classes:", train_generator.class_indices)

# -----------------------------
# SAVE CLASS INDICES (FIXED)
# -----------------------------
pkl_path = os.path.join(PROJECT_DIR, "class_indices.pkl")

with open(pkl_path, "wb") as f:
    pickle.dump(train_generator.class_indices, f)

print("class_indices.pkl saved at:", pkl_path)

# OPTIONAL: VERIFY PICKLE LOAD
with open(pkl_path, "rb") as f:
    loaded_classes = pickle.load(f)

print("Loaded classes from pickle:", loaded_classes)

# -----------------------------
# VGG19 MODEL
# -----------------------------
base_model = VGG19(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_generator, epochs=6)

# -----------------------------
# SAVE MODEL
# -----------------------------
model_path = os.path.join(PROJECT_DIR, "dogbreed.h5")
model.save(model_path)

print("Model saved at:", model_path)
print("Training Completed Successfully ✅")
