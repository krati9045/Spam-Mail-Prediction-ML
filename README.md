# Spam Mail Prediction using Machine Learning

## Problem Statement

The objective of this project is to develop a machine learning model that classifies messages as **Spam** or **Ham (legitimate)** based on their text content.

This is a **binary classification** problem where each message is assigned to one of two categories.

## Dataset

The dataset contains **5,572 messages** with two columns:

* **Category** - Class label (`spam` or `ham`)
* **Message** - Text content of the message

### Target

The `Category` column was label encoded as:

* `0` - Spam
* `1` - Ham

## Solution Approach

The project follows an end-to-end text classification workflow:

**Data Preprocessing -> Label Encoding -> Train-Test Split -> TF-IDF Feature Extraction -> Model Training -> Evaluation -> Prediction**

The text data was converted into numerical feature vectors using **TF-IDF Vectorization** with English stop words removed and lowercase conversion enabled.

A **Logistic Regression** model was then trained on the extracted features.

## Observations

The dataset contains significantly more ham messages than spam messages, with **4,825 ham messages** and **747 spam messages**.

TF-IDF provided a numerical representation of the message text that could be used effectively by the classification model.

## Model Performance

| Dataset       |   Accuracy |
| ------------- | ---------: |
| Training Data | **96.77%** |
| Testing Data  | **96.68%** |

The trained model was also tested on a new sample message and classified it as **Spam**.

## Findings

The Logistic Regression model achieved **96.68% accuracy on the testing data**, showing strong classification performance on the held-out messages.

## Conclusion

This project successfully implemented an end-to-end spam message classification workflow, from text preprocessing and feature extraction to model training, evaluation, and prediction on new messages.

It provided practical experience with **text preprocessing, label encoding, TF-IDF feature extraction, Logistic Regression, and binary classification**.

## Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* TF-IDF Vectorization
* Logistic Regression
* Google Colab / Jupyter Notebook
