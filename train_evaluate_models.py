import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
from tensorflow.keras.applications import ResNet50, InceptionV3, VGG16, MobileNetV2, DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Define constants
batch_size = 32
input_shape = (224, 224, 3)

# Load data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '/kaggle/input/cataract-image-dataset/processed_images/train/',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    '/kaggle/input/cataract-image-dataset/processed_images/test/',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary'
)

# Define models
def create_model_convnet():
    """Create a ConvNet model"""
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_model_vgg16():
    """Create a VGG16 model"""
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_model_resnet50():
    """Create a ResNet50 model"""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_model_inceptionv3():
    """Create an InceptionV3 model"""
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    for layer in base_model.layers:
        layer.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_model_mobilenetv2():
    """Create a MobileNetV2 model"""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_model_densenet121():
    """Create a DenseNet121 model"""
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate models
def train_and_evaluate_model(model, model_name):
    """Train and evaluate a given model"""
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=test_generator
    )

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Training and Validation Accuracy ({model_name})')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss ({model_name})')
    plt.legend()

    plt.show()

    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f'Test Accuracy ({model_name}): {test_accuracy * 100:.2f}%')

# Train and evaluate models
models = [
    create_model_convnet,
    create_model_vgg16,
    create_model_resnet50,
    create_model_inceptionv3,
    create_model_mobilenetv2,
    create_model_densenet121
]

model_names = [
    'ConvNet',
    'VGG16',
    'ResNet50',
    'InceptionV3',
    'MobileNetV2',
    'DenseNet121'
]

for i, model_func in enumerate(models):
    model = model_func()
    train_and_evaluate_model(model, model_names[i])

# Save the models
for i, model_func in enumerate(models):
    model = model_func()
    model.save(f'model_{model_names[i]}.h5')

print("Models saved successfully!")

# Load the models
loaded_models = []
for i, model_name in enumerate(model_names):
    loaded_model = load_model(f'model_{model_name}.h5')
    loaded_models.append(loaded_model)

print("Models loaded successfully!")

# Make predictions on the test set
predictions = []
for model in loaded_models:
    pred = model.predict(test_generator)
    predictions.append(pred)

print("Predictions made successfully!")

# Evaluate the models
for i, pred in enumerate(predictions):
    pred_class = (pred > 0.5).astype(int)
    accuracy = accuracy_score(test_labels, pred_class)
    print(f'Model {model_names[i]} Accuracy: {accuracy * 100:.2f}%')

print("Evaluation complete!")