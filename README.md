# 🩻 Chest X-Ray Pneumonia Detection

This project focuses on building a deep learning model to classify chest X-ray images as **NORMAL** or **PNEUMONIA** using a CNN-based binary classifier.

## 📁 Dataset

The dataset is the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) from Kaggle, which includes:

- Training set: `chest_xray/train/`
- Validation set: `chest_xray/val/`
- Test set: `chest_xray/test/`

Each folder contains two subdirectories:  
- `NORMAL`  
- `PNEUMONIA`

## 🧠 Model Architecture

The model is a **Convolutional Neural Network (CNN)** implemented in Keras:

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```
Loss function: binary_crossentropy

Optimizer: Adam

Final Activation: sigmoid for binary classification

Evaluation Metric: accuracy, confusion_matrix, classification_report

## 📌 Notes
Validation set is very small (16 images), so results may vary significantly.

You can improve the model further with:

* Transfer learning (e.g., MobileNet, EfficientNet)

* More aggressive augmentation

