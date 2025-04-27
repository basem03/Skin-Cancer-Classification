# Skin Cancer Classification and Segmentation
# Description
This repository hosts a comprehensive deep learning project developed using PyTorch for skin cancer classification and lesion segmentation from dermoscopic images. The project implements two primary tasks: classifying skin lesions into seven distinct categories (e.g., melanoma, basal cell carcinoma) using a ResNet-18-based model and segmenting lesion regions using a U-Net architecture with a ResNet-18 encoder. Additionally, it includes a multi-task model that performs both classification and segmentation simultaneously, leveraging shared features for efficiency. The codebase is designed to handle dataset preprocessing, model training, evaluation, and visualization, making it a valuable resource for researchers, data scientists, and medical imaging enthusiasts interested in applying deep learning to dermatology.
The project emphasizes reproducibility with fixed random seeds, robust data preprocessing (e.g., handling corrupt files, stratified data splits), and detailed evaluation metrics (e.g., classification reports, IoU/Dice scores, visualizations). It is intended for users with access to a compatible dataset, such as those from the International Skin Imaging Collaboration (ISIC) or similar sources.

# Features

**Classification Model**: Utilizes a pre-trained ResNet-18 model, fine-tuned to classify skin lesions into seven categories with a custom fully connected layer.
**Segmentation Model**: Implements a U-Net architecture with a ResNet-18 encoder for binary lesion segmentation, producing pixel-wise masks.
**Multi-Task Model**: Combines classification and segmentation in a single model with a shared ResNet-18 encoder and separate heads for each task.
**Data Preprocessing**: Includes robust checks for corrupt images/masks, stratified train/validation/test splits, and image transformations (e.g., resizing, normalization).
**Evaluation Metrics**: Generates detailed classification reports (precision, recall, F1-score), confusion matrices, segmentation metrics (IoU, Dice), and visualizations (predicted masks, training curves).
**Reproducibility**: Sets random seeds (torch.manual_seed(42), np.random.seed(42)) for consistent results across runs.
**GPU Support**: Automatically detects and utilizes CUDA-enabled GPUs if available, with fallback to CPU.


# Classes
The dataset includes seven skin lesion categories:

Melanoma
Nevus
Basal Cell Carcinoma
Actinic Keratosis
Benign Keratosis
Dermatofibroma
Vascular Lesion


# Prerequisites

Python: Version 3.8 or higher.
PyTorch: Version 1.9 or higher, with optional CUDA support for GPU acceleration.
Hardware: GPU recommended for faster training, but CPU is supported.
Dependencies: Listed in requirements.txt for reproducibility.



# Preprocessing:
Loads GroundTruth.csv and converts one-hot encoded labels to class names.
Checks for and removes duplicate entries.
Splits data into train (70%), validation (15%), and test (15%) sets with stratified sampling.
Verifies image and mask integrity, removing corrupt or missing files.
Creates separate datasets for classification (images only) and segmentation (images + masks).


# Training:
Trains a classification model (ResNet-18, 15 epochs, batch size 32) using CrossEntropyLoss and Adam optimizer (lr=1e-4).
Trains a segmentation model (U-Net, 25 epochs, batch size 16) using BCEWithLogitsLoss and Adam optimizer (lr=1e-4).
Trains a multi-task model (10 epochs, batch size 16) combining both tasks with a shared ResNet-18 encoder.
Saves the best models based on validation loss (best_classification_model.pth, best_segmentation_model.pth).


# Evaluation:
Generates classification reports (precision, recall, F1-score) and confusion matrices for the test set.
Computes segmentation metrics (mean IoU, Dice coefficient) and visualizes predicted masks.
Evaluates the multi-task model for both accuracy and segmentation metrics.
Plots training curves for loss and accuracy.




# Outputs:

Saved Models: best_classification_model.pth, best_segmentation_model.pth.
Classification Results: Detailed report with per-class metrics and a confusion matrix.
Segmentation Results: Mean IoU and Dice scores, visualizations of original images, ground truth masks, predicted masks, and overlays.
Multi-Task Results: Classification accuracy, IoU, and Dice scores for the test set.
Visualizations: Training loss/accuracy curves saved as plots.


# Customization:

Modify hyperparameters in main.py (e.g., learning rate, batch size, number of epochs) to optimize performance.
Adjust dataset paths or file naming conventions to match your dataset.
Extend the code to include additional models, loss functions, or evaluation metrics as needed.



# Model Details
**Classification Model**

Architecture: Pre-trained ResNet-18 with a modified fully connected layer to output 7 classes.
Input: RGB images resized to 224x224, normalized (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
Loss Function: CrossEntropyLoss.
Optimizer: Adam (lr=1e-4).
Training: 15 epochs, batch size 32.

**Segmentation Model**

Architecture: U-Net with a ResNet-18 encoder, decoder blocks with upsampling, and skip connections.
Input: RGB images (224x224, normalized) and binary masks (224x224, nearest-neighbor interpolation).
Output: Single-channel binary mask (sigmoid activation).
Loss Function: BCEWithLogitsLoss.
Optimizer: Adam (lr=1e-4).
Training: 25 epochs, batch size 16.

**Multi-Task Model**

Architecture: Shared ResNet-18 encoder with two heads:
Classification head: Adaptive pooling, linear layer for 7 classes.
Segmentation head: Decoder blocks for binary mask output.


Input: Same as individual models.
Loss Function: Combined CrossEntropyLoss (classification) and BCEWithLogitsLoss (segmentation).
Optimizer: Adam (lr=1e-4).
Training: 10 epochs, batch size 16.

# Evaluation
The project provides comprehensive evaluation for both tasks:

# Classification:
Metrics: Precision, recall, F1-score per class, overall accuracy.
Visualization: Confusion matrix with class-wise performance.


# Segmentation:
Metrics: Mean Intersection over Union (IoU) and Dice coefficient.
Visualization: Side-by-side comparison of original images, ground truth masks, predicted masks, and overlays.


# Multi-Task Model:
Metrics: Classification accuracy, IoU, and Dice for segmentation.


# Training Curves:
Plots of training and validation loss for both models.
Validation accuracy for classification.


# Contact
For questions, feedback, or collaboration, please reach out:

Email: basemhesham200318@gmail.com



