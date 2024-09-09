# CODTECH-TASK4

**Name**: Ravi Kiran B A\
**Company**: CODTECH IT SOLUTIONS\
**ID**: CT08DS2280\
**Domain**: Artificial Intelligence\
**Duration**: August to October 2024

# Overview of the Project

### Project: Rice Image Classification with Deep Learning

### Get the dataset from: https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset

## Objective
The objective of this project is to develop and evaluate deep learning models for classifying rice images into different categories. 
This includes training a custom Convolutional Neural Network (CNN) and fine-tuning a pre-trained MobileNetV2 model to achieve high accuracy in rice image classification.

## Key Activities
Data Preparation
Unzipped and extracted the rice image dataset.
Set up image data generators for training and validation.

Model Development
Built and trained a custom CNN model for image classification.
Fine-tuned a pre-trained MobileNetV2 model for improved performance.

Model Evaluation
Evaluated the performance of both the custom CNN and fine-tuned MobileNetV2 models.
Plotted training history, confusion matrices, and class distribution of predictions.

Prediction and Visualization
Implemented a function to predict the class of new images.
Visualized training and evaluation results using plots.

Model Saving
Saved the trained models and vectorizer for future use.

## Technologies Used

Programming Languages: Python

Python Libraries: TensorFlow, Keras, scikit-learn, NumPy, Pandas, Matplotlib, Seaborn
Deep Learning Frameworks: TensorFlow/Keras
Data Handling: Pandas for data manipulation and NumPy for numerical operations
Visualization: Matplotlib and Seaborn for plotting

## Files
Rice_Image_Dataset.zip: Contains the rice image dataset.

## Usage

Setup
Ensure you have all the required Python libraries installed. Use pip install -r requirements.txt to install dependencies.

Run the Script
Execute main.py to perform data preparation, model training, and evaluation.

Prediction
Use the predict_image(img_path) function to classify new images by providing the path to the image.

## Features
Image classification using a custom CNN model.
Enhanced performance with a fine-tuned MobileNetV2 model.
Visualization of training history, confusion matrices, and class distributions.
Model and vectorizer saving for future use.

## Conclusion
The project successfully demonstrates the application of deep learning techniques to the classification of rice images. 
By comparing a custom-built CNN model with a fine-tuned MobileNetV2 model, we were able to achieve effective classification results. 
The fine-tuned MobileNetV2 model, in particular, showed improved accuracy, demonstrating the effectiveness of transfer learning for this task.

## Acknowledgements
TensorFlow/Keras: For providing deep learning frameworks and pre-trained models.
scikit-learn: For tools and utilities for model evaluation and hyperparameter tuning.
Matplotlib and Seaborn: For visualization of results.
Pandas and NumPy: For data manipulation and numerical operations.
