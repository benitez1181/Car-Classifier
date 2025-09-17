from pathlib import Path
import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

def move_class(name,brands):
  os.mkdir(name)
  new_directory = Path(name)

  for dirpath, dirname, filenames in os.walk(data/ "training_set"):
    if os.path.basename(dirpath) in brands:
      shutil.move(dirpath, new_directory)

  return new_directory

brands = ["Toyota", "Audi", "Hyundai", "Nissan", "BMW", "Chevrolet", "Ford", "Volkswagen", "Dodge", "Mercedes"]
training = move_class("new_training", brands)

# Data augmentation
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(0.2)
])

# Building the Model 
input = tf.keras.Input(shape=(200,200,3))
x = augmentation(input)
rescaling = layers.Rescaling(1./255)(x) # Normalizes the pixels in the range of 0-1
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(rescaling)
x = layers.MaxPooling2D(pool_size=(2,2))(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)
x = layers.Conv2D(filters=512, kernel_size=(2,2), activation='relu')(x)
x = layers.Conv2D(filters=512, kernel_size=(2,2), activation='relu')(x)
x = layers.Flatten()(x)
x= layers.Dense(50, activation='relu')(x)
outputs = layers.Dense(10, activation="softmax")(x)

my_model = tf.keras.Model(inputs=input, outputs=outputs)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath="convnet_from_scratch.keras",
                                       save_best_only=True,
                                       monitor="val_loss")]

my_model.compile(
    optimizer= tf.keras.optimizers.SGD(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

from tensorflow.keras.utils import image_dataset_from_directory

train_data = image_dataset_from_directory(
    data / "training_set",
    image_size=(220, 220),
    batch_size=32,
    label_mode="categorical",
    validation_split=.20,
    subset='training',
    seed=42,
    shuffle=True)

val_data = image_dataset_from_directory(
    data / "training_set",
    image_size=(220, 220),
    batch_size=32,
    label_mode="categorical",
    validation_split=.20,
    subset='validation',
    seed=42,
    shuffle=True)

#Training the model 

results = my_model.fit(
    train_data,
    epochs=70,
    callbacks=callbacks,
    validation_data=val_data
)

# Showing training progress

import matplotlib.pyplot as plt

accuracy = results.history["accuracy"]
val_accuracy = results.history["val_accuracy"]
loss = results.history["loss"]
val_loss = results.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()
