# Cloudy-Eyes

This repository contains code for training and evaluating various deep learning models to classify cataract images. The models used include ConvNet, VGG16, ResNet50, InceptionV3, MobileNetV2, and DenseNet121. The primary aim is to aid in the automated diagnosis of cataracts using image classification techniques.

## Dataset

The dataset used in this project comprises images of eyes with and without cataracts. The images are divided into training and test sets.
Each set contains images that are resized to 224x224 pixels and are categorized into binary classes: cataract and non-cataract.

## Need for Automated Cataract Diagnosis

Cataracts are a leading cause of blindness worldwide. Early detection and treatment are crucial for preventing vision loss. Automated diagnosis using deep learning can assist healthcare professionals by:

- Providing rapid and accurate diagnostics.
- Reducing the workload on ophthalmologists.
- Enabling large-scale screening programs in underserved areas.

## Models

The project implements several deep learning models for cataract image classification. Each model is described below:

### 1. Convolutional Neural Network (ConvNet)

A simple ConvNet architecture with the following layers:

- Conv2D and MaxPooling layers
- Dense layers with dropout for regularization
- Output layer with sigmoid activation

### 2. VGG16

A pre-trained VGG16 model with the following modifications:

- Pre-trained on ImageNet, excluding the top layers
- Flattening and adding dense layers
- Output layer with sigmoid activation

### 3. ResNet50

A pre-trained ResNet50 model with the following modifications:

- Pre-trained on ImageNet, excluding the top layers
- Global average pooling and adding dense layers
- Output layer with sigmoid activation

### 4. InceptionV3

A pre-trained InceptionV3 model with the following modifications:

- Pre-trained on ImageNet, excluding the top layers
- Global average pooling and adding dense layers
- Output layer with sigmoid activation

### 5. MobileNetV2

A pre-trained MobileNetV2 model with the following modifications:

- Pre-trained on ImageNet, excluding the top layers
- Global average pooling and adding dense layers
- Output layer with sigmoid activation

### 6. DenseNet121

A pre-trained DenseNet121 model with the following modifications:

- Pre-trained on ImageNet, excluding the top layers
- Global average pooling and adding dense layers
- Output layer with sigmoid activation

## Training and Evaluation

Each model is trained for 10 epochs using binary cross-entropy loss and the Adam optimizer. Training and validation accuracies and losses are plotted for each epoch to monitor performance.

The models are evaluated on the test set, and their accuracy is reported. The models are saved in HDF5 format for future use.

## Usage

### Dependencies

Ensure you have the following libraries installed:

- numpy
- pandas
- tensorflow
- matplotlib
- scikit-learn

### Training and Evaluating Models

To train and evaluate the models, run:

```python
python train_evaluate_models.py
```

This script will train each model, plot training and validation accuracies and losses, and save the models.

## Results

The results of the models will be printed as test accuracies after training and evaluation. Additionally, plots of training and validation accuracy and loss for each model will be displayed.


## Conclusion

This repository demonstrates the application of various deep learning models for cataract image classification. The automated approach can potentially assist in the early detection and treatment of cataracts, contributing to better eye care and prevention of blindness.

