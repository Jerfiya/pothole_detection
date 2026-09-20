# Pothole Detection Using CNN

A computer vision project that classifies road images as **Normal** or **Pothole** using a Convolutional Neural Network (CNN).

## 📌 Project Overview

Potholes can cause accidents, vehicle damage, and unsafe road conditions. This project uses image classification to automatically identify whether a road image contains a pothole.

The model takes a road image as input and predicts one of two classes:

* **Normal**
* **Pothole**

The project was developed using Python and TensorFlow/Keras without using OpenCV.

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pillow (PIL)
* Scikit-learn
* Matplotlib
* Git & GitHub

## 🧠 Model Architecture

The project uses a Convolutional Neural Network with:

1. Input image size: **128 × 128 × 3**
2. Rescaling layer
3. Convolutional layer – 32 filters
4. Max Pooling
5. Convolutional layer – 64 filters
6. Max Pooling
7. Convolutional layer – 128 filters
8. Max Pooling
9. Flatten layer
10. Dense layer – 128 neurons
11. Dropout – 0.5
12. Output layer – Sigmoid activation

The model contains approximately **3.3 million trainable parameters**.

## 📂 Project Structure

```text
pothole_detection/
│
├── data/
│   ├── train/
│   │   ├── Normal/
│   │   └── Pothole/
│   ├── val/
│   │   ├── Normal/
│   │   └── Pothole/
│   └── test/
│
├── model.py
├── train.py
├── test.py
├── evaluation.py
├── predict.py
├── pothole_model.keras
├── README.md
└── .gitignore
```

> The dataset is excluded from the Git repository using `.gitignore`.

## 📊 Dataset

The training dataset contains two classes:

* Normal road images
* Pothole road images

The dataset used for training contained **1,157 images**, while the validation set contained **108 images**.

A separate test set containing **136 images** was used to evaluate the trained model.

## ⚙️ Training

The images are resized to **128 × 128 pixels** before being given to the CNN.

The model was trained for **15 epochs** using:

* Optimizer: Adam
* Loss function: Binary Crossentropy
* Batch size: 32
* Activation: ReLU for hidden layers
* Output activation: Sigmoid

## 📈 Model Performance

### Test Results

The model achieved approximately **90.44% accuracy** on the test set.

| Metric    | Normal | Pothole |
| --------- | -----: | ------: |
| Precision |   0.97 |    0.85 |
| Recall    |   0.85 |    0.97 |
| F1-score  |   0.90 |    0.91 |

### Confusion Matrix

```text
                 Predicted
                 Normal  Pothole

Actual Normal       60      11
Actual Pothole       2      63
```

This means:

* 60 normal images were correctly classified.
* 63 pothole images were correctly classified.
* 11 normal images were incorrectly classified as potholes.
* 2 pothole images were incorrectly classified as normal.

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Jerfiya/pothole_detection.git
cd pothole_detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install tensorflow numpy pillow scikit-learn matplotlib
```

### 5. Make a prediction

Run:

```bash
python predict.py
```

The program will ask for the path of an image:

```text
Enter the path of the road image:
```

It will then display a prediction such as:

```text
Prediction: POTHOLE
Confidence: 74.44 %
```

## 🔄 Project Workflow

```text
Road Image
    ↓
Image Preprocessing
    ↓
Resize to 128 × 128
    ↓
CNN
    ↓
Feature Extraction
    ↓
Classification
    ↓
Normal / Pothole
```

## 🎯 Future Improvements

Possible improvements include:

* Creating a web interface for image upload
* Increasing the size and diversity of the dataset
* Applying data augmentation
* Experimenting with different CNN architectures
* Improving performance on different road conditions
* Deploying the model as a web application

## 👩‍💻 Author

**Jerfiya**

GitHub: https://github.com/Jerfiya
