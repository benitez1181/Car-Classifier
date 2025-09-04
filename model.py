from pathlib import Path
import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# Building the Model 
inputs = layers.Input(shape=(220,220,3))
rescaling = layers.Rescaling(1./255)(inputs) # Normalizes the pixels in the range of 0-1
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(rescaling)
x = layers.MaxPooling2D(pool_size=(2,2))(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(17, activation="softmax")(x)

my_model = tf.keras.Model(inputs=inputs, outputs=outputs)

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
