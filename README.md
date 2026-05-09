# Image Classification Comparative Study

This project presents a comparative study of image classification using three different machine learning models: **Naive Bayes**, **Decision Tree**, and **Multi-Layer Perceptron (MLP)**. The goal is to evaluate and compare the performance of these models on a dataset containing images of different classes.

## 👥 Authors
* **Anwar Atawna** - 1222275
* **Qusai Abu Sonds** - 1221082

## 📝 Project Overview
The project implements a full machine learning pipeline, including:
1.  **Data Loading**: Reading images from local directories and resizing them to a uniform size (32x32).
2.  **Preprocessing**: Normalizing pixel values to the range [0, 1].
3.  **Dimensionality Reduction**: Applying **Principal Component Analysis (PCA)** to reduce the feature space while retaining significant variance (50 components).
4.  **Classification**: Training and testing three models:
    *   **Gaussian Naive Bayes**
    *   **Decision Tree Classifier**
    *   **Feedforward Neural Network (MLP)**
5.  **Evaluation**: Using metrics such as Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.

## 📂 Dataset Structure
The project expects a dataset directory named `dataset1/dataset1` with the following structure:
```text
dataset1/
└── dataset1/
    ├── bird/
    ├── dog/
    └── flower/
```
Each subdirectory contains image files corresponding to its label.

## 🛠️ Prerequisites
Ensure you have the following Python libraries installed:
*   `opencv-python` (cv2)
*   `numpy`
*   `matplotlib`
*   `scikit-learn`

You can install them using pip:
```bash
pip install opencv-python numpy matplotlib scikit-learn
```

## 🚀 How to Run
1.  Place your dataset in the `dataset1/` folder as described in the structure above.
2.  Run the main script:
    ```bash
    python Project2Ai.py
    ```

## 📊 Results and Evaluation
The script outputs the performance metrics for each model:
*   **Accuracy**: Overall correctness of the model.
*   **Classification Report**: Detailed breakdown of precision, recall, and f1-score per class.
*   **Confusion Matrix**: Visualization of misclassifications.
*   **Decision Tree Visualization**: A plot showing the top levels of the decision tree.

## 📜 Summary of Findings
The results are summarized at the end of the execution, providing a quick comparison of the accuracies achieved by Naive Bayes, Decision Tree, and the MLP model.
