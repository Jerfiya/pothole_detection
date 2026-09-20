import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# 1. Load the trained model
# -----------------------------

model = tf.keras.models.load_model("pothole_model.keras")

# -----------------------------
# 2. Test image settings
# -----------------------------

image_size = (128, 128)
test_path = Path("data/test")

actual_labels = []
predicted_labels = []

# -----------------------------
# 3. Read every test image
# -----------------------------

supported_extensions = ["*.jpg", "*.jpeg", "*.png"]

image_paths = []

for extension in supported_extensions:
    image_paths.extend(test_path.glob(extension))

print("Total test images:", len(image_paths))

for image_path in image_paths:

    # Open image using PIL
    image = Image.open(image_path).convert("RGB")

    # Resize image
    image = image.resize(image_size)

    # Convert image to NumPy array
    image_array = np.array(image, dtype=np.float32)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # -----------------------------
    # 4. Make prediction
    # -----------------------------

    prediction = model.predict(image_array, verbose=0)[0][0]

    if prediction >= 0.5:
        predicted_label = 1
    else:
        predicted_label = 0

    # -----------------------------
    # 5. Get actual label
    # -----------------------------

    if image_path.name.lower().startswith("pothole"):
        actual_label = 1
    else:
        actual_label = 0

    actual_labels.append(actual_label)
    predicted_labels.append(predicted_label)

# -----------------------------
# 6. Calculate accuracy
# -----------------------------

accuracy = accuracy_score(actual_labels, predicted_labels)

print("\nTest Accuracy:", accuracy)

# -----------------------------
# 7. Confusion Matrix
# -----------------------------

matrix = confusion_matrix(
    actual_labels,
    predicted_labels
)

print("\nConfusion Matrix:")
print(matrix)

# -----------------------------
# 8. Classification Report
# -----------------------------

print("\nClassification Report:")

print(
    classification_report(
        actual_labels,
        predicted_labels,
        target_names=["Normal", "Pothole"]
    )
)