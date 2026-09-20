import tensorflow as tf
import matplotlib.pyplot as plt
from model import model

# -----------------------------
# 1. Basic settings
# -----------------------------

image_size = (128, 128)
batch_size = 32

# -----------------------------
# 2. Dataset locations
# -----------------------------

training_path = "data/train"
validation_path = "data/val"

# -----------------------------
# 3. Load training dataset
# -----------------------------

training_dataset = tf.keras.utils.image_dataset_from_directory(
    training_path,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True
)

# -----------------------------
# 4. Load validation dataset
# -----------------------------

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_path,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=False
)

# -----------------------------
# 5. Check classes
# -----------------------------

print("Classes:", training_dataset.class_names)

# -----------------------------
# 6. Train the model
# -----------------------------

history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=15
)

# -----------------------------
# 7. Save the trained model
# -----------------------------

model.save("pothole_model.keras")

print("Model training completed!")
print("Model saved as pothole_model.keras")
# -----------------------------
# 8. Plot training and validation accuracy
# -----------------------------

plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.show()


# -----------------------------
# 9. Plot training and validation loss
# -----------------------------

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.show()