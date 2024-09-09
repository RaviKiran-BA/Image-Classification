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
- **Data Preparation**
  - Unzipped and extracted the rice image dataset.
  - Set up image data generators for training and validation.

- **Model Development**
  - Built and trained a custom CNN model for image classification.
  - Fine-tuned a pre-trained MobileNetV2 model for improved performance.

- **Model Evaluation**
  - Evaluated the performance of both the custom CNN and fine-tuned MobileNetV2 models.
  - Plotted training history, confusion matrices, and class distribution of predictions.

- **Prediction and Visualization**
  - Implemented a function to predict the class of new images.
  - Visualized training and evaluation results using plots.

- **Model Saving**
  - Saved the trained models and vectorizer for future use.


## Technologies Used

Programming Languages: Python

- **Programming Languages**
  - Python
- **Python Libraries**
  - TensorFlow, Keras, scikit-learn, NumPy, Pandas, Matplotlib, Seaborn
- **Deep Learning Frameworks**
  - TensorFlow/Keras
- **Data Handling**
  - Pandas for data manipulation, NumPy for numerical operations
- **Visualization**
  - Matplotlib and Seaborn for plotting

## Files
- **Rice_Image_Dataset.zip**
  - Contains the rice image dataset.
- **main.py**
  - The main Python script for data preprocessing, model training, and evaluation.

## Features
- Image classification using a custom CNN model.
- Enhanced performance with a fine-tuned MobileNetV2 model.
- Visualization of training history, confusion matrices, and class distributions.
- Model and vectorizer saving for future use.

## Output

### Accuracy

![Accuracy](https://github.com/user-attachments/assets/57b47484-6bf0-492f-87f9-1901ecb4c943)

### Confusion Matrix

![Confusion_Matrix](https://github.com/user-attachments/assets/ac534792-5761-4883-adaf-824030e489b8)

### Class Distributions of Predictions

![Class_Distribution_of_Predictions](https://github.com/user-attachments/assets/bfb89796-bfd2-40b5-b322-433b71d5ab9d)

## Conclusion
The project successfully demonstrates the application of deep learning techniques to the classification of rice images. 
By comparing a custom-built CNN model with a fine-tuned MobileNetV2 model, we were able to achieve effective classification results. 
The fine-tuned MobileNetV2 model, in particular, showed improved accuracy, demonstrating the effectiveness of transfer learning for this task.

## Acknowledgements
- **TensorFlow/Keras**
  - For providing deep learning frameworks and pre-trained models.
- **scikit-learn**
  - For tools and utilities for model evaluation and hyperparameter tuning.
- **Matplotlib and Seaborn**
  - For visualization of results.
- **Pandas and NumPy**
  - For data manipulation and numerical operations.
