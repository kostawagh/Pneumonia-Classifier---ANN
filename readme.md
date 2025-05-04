# Pneumonia Detection Using Classical ML & ANN

## Project Overview
This project implements a classical machine learning pipeline and a simple Artificial Neural Network (ANN) to detect pneumonia from chest X-ray images, without relying on deep convolutional neural networks (CNNs). We extract handcrafted shape and texture features from X-ray images and train both traditional ML models (SVM, Random Forest, KNN, Logistic Regression) and an ANN to compare their performance.

## Features
- **Image Preprocessing**: Grayscale conversion, resizing, histogram visualization  
- **Image Enhancement**: Histogram Equalization, CLAHE, Gamma Correction  
- **Image Filtering**: Gaussian Blur, Median Filter, Sobel Edge Detection  
- **Image Segmentation**: Otsu Thresholding, Morphological Opening/Closing  
- **Feature Extraction**:
  - Shape features (area, perimeter, aspect ratio, compactness)  
  - Texture features (GLCM: contrast, dissimilarity, homogeneity, energy, correlation)  
- **Modeling**:
  - Traditional ML: SVM, Random Forest, KNN, Logistic Regression  
  - ANN: Two hidden layers with dropout and early stopping  
- **Evaluation**: Validation/Test accuracy, loss & accuracy curves, confusion matrix, classification report  
