import numpy as np
from PIL import Image
import tensorflow as tf

# -----------------------------
# 1. Load the trained model
# -----------------------------

model = tf.keras.models.load_model("pothole_model.keras")

# -----------------------------
# 2. Ask the user for an image
# -----------------------------

image_path = input("Enter the path of the road image: ")

# -----------------------------
# 3. Open the image
# -----------------------------

try:
    image = Image.open(image_path).convert("RGB")

except FileNotFoundError:
    print("Image not found. Please check the path.")
    exit()

# -----------------------------
# 4. Resize the image
# -----------------------------

image = image.resize((128, 128))

# -----------------------------
# 5. Convert to NumPy array
# -----------------------------

image_array = np.array(image, dtype=np.float32)

# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# -----------------------------
# 6. Make prediction
# -----------------------------

prediction = model.predict(image_array, verbose=0)[0][0]

# -----------------------------
# 7. Display result
# -----------------------------

if prediction >= 0.5:

    confidence = float(prediction) * 100

    print("\nPrediction: POTHOLE")
    print("Confidence:", round(confidence, 2), "%")

else:

    confidence = (1 - float(prediction)) * 100

    print("\nPrediction: NORMAL")
    print("Confidence:", round(confidence, 2), "%")