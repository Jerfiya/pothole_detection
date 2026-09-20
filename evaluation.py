import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Your confusion matrix from the test results
confusion_matrix = np.array([
    [60, 11],
    [2, 63]
])

# Class names
class_names = ["Normal", "Pothole"]

# Create the graph
display = ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix,
    display_labels=class_names
)

display.plot()

plt.title("Pothole Detection - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.show()
# Test metrics
metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score"
]

metric_values = [
    0.9044,
    0.85,
    0.97,
    0.91
]

plt.figure(figsize=(8, 5))

plt.bar(metric_names, metric_values)

plt.ylim(0, 1)

plt.ylabel("Score")
plt.title("Pothole Detection Model Performance")

plt.show()