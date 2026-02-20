import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import json
import os



IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

train_dir = "dataset/archive/dataset2-master/dataset2-master/images/train"
test_dir = "dataset/archive/dataset2-master/dataset2-master/images/test"


# DATA GENERATORS
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# Save class indices
with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)

print("Class Indices:", train_generator.class_indices)

# MODEL (Transfer Learning)

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(4, activation="softmax")
])

# COMPILE
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
# TRAIN
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# SAVE MODEL

model.save("blood_cell_model.h5")

print("Training complete. Model saved successfully.")
