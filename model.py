import tensorflow as tf

# Image size
image_size = (128, 128)

# Create the CNN model
model = tf.keras.Sequential([

    tf.keras.Input(shape=(128, 128, 3)),

    tf.keras.layers.Rescaling(1./255),

    # First convolution layer
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    # Second convolution layer
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    # Third convolution layer
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    # Convert feature maps into one long list
    tf.keras.layers.Flatten(),

    # Learn patterns from the extracted features
    tf.keras.layers.Dense(128, activation="relu"),

    # Reduce overfitting
    tf.keras.layers.Dropout(0.5),

    # Final output: 0 = Normal, 1 = Pothole
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Configure the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Display model structure
model.summary()