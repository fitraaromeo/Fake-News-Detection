# Fake News Detection using LSTM and BERT

# 📌 Project Description
This project is developed as part of the Final Practical Exam (Ujian Akhir Praktikum / UAP) for the Machine Learning course.
The objective of this project is to build a fake news detection system using text-based machine learning models, compare the performance between non-pretrained and pretrained (transfer learning) models, and deploy the trained models into a simple web application using Streamlit.

## 📂 Dataset
- Dataset Name: Fake and Real News Dataset
- Source: Kaggle https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

## Dataset Description
The dataset consists of two classes:
1. Fake News (label = 1) → misleading or hoax news articles
2. Real News (label = 0) → factual and verified news articles

Each data sample contains:
1. title : headline of the news article
2. text : full news content

The dataset is merged and shuffled before preprocessing.

## 🧹 Data Preprocessing
The following preprocessing steps are applied:
1. Combining title and text into a single text field
2. Converting text to lowercase
3. Removing URLs
4. Removing non-alphabet characters
5. Removing extra whitespace

The dataset is split into:
1. 80% Training data
2. 10% Validation data
3. 10% Testing data

## 🧠 Models Implemented
This project implements three machine learning models as required by the UAP guidelines:
1. LSTM (Non-Pretrained Model)
- Built from scratch using Recurrent Neural Network architecture
- Used as a baseline model for text classification

2. BERT (Pretrained Model – Transfer Learning)
- Utilizes a pretrained BERT model
- Fine-tuned on the fake news dataset for classification

3. DistilBERT (Pretrained Model – Transfer Learning)
- A lighter and faster version of BERT
- Used to compare performance and efficiency against BERT

## Tech Stack
- Python
- TensorFlow / PyTorch
- HuggingFace Transformers
- Streamlit