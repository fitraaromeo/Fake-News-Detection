# 📰 Fake News Detection using LSTM, BERT, and DistilBERT

## 📌 Project Description
This project is developed as part of the **Final Practical Exam (Ujian Akhir Praktikum / UAP)** for the **Machine Learning** course.

The objective of this project is to build a **fake news detection system** using **text-based machine learning models**, compare the performance between **non-pretrained models** and **pretrained (transfer learning) models**, and deploy the trained models into a **web-based application using Streamlit**.

---

## 📂 Dataset
- **Dataset Name:** Fake and Real News Dataset  
- **Source:** Kaggle  
  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset  

### Dataset Description
The dataset consists of two classes:
- **Fake News (label = 1):** misleading or hoax news articles  
- **Real News (label = 0):** factual and verified news articles  

Each data sample contains:
- `title` : headline of the news article  
- `text`  : full news content  

The dataset is merged and shuffled before preprocessing.

---

## 🧹 Data Preprocessing
The following preprocessing steps are applied:
1. Combining `title` and `text` into a single text field
2. Converting text to lowercase
3. Removing URLs
4. Removing non-alphabet characters
5. Removing extra whitespace

The dataset is split into:
- **80% Training data**
- **10% Validation data**
- **10% Testing data**

---

## 🧠 Models Implemented
This project implements **three machine learning models** as required by the UAP guidelines:

### 1️⃣ LSTM (Non-Pretrained Model)
- Built from scratch using Recurrent Neural Network (LSTM) architecture
- Serves as a baseline model for text classification

### 2️⃣ BERT (Pretrained Model – Transfer Learning)
- Utilizes a pretrained **BERT-base-uncased** model
- Fine-tuned on the fake news dataset for binary classification

### 3️⃣ DistilBERT (Pretrained Model – Transfer Learning)
- A lighter and faster variant of BERT
- Used to compare performance and efficiency against BERT

---

## 📊 Model Evaluation
All models are evaluated using the **test dataset** with the following metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### 📈 Evaluation Results (Test Set)

| Model        | Accuracy | Precision | Recall | F1-score | Support |
|--------------|----------|-----------|--------|----------|---------|
| LSTM         | 1.00     | 1.00      | 1.00   | 1.00     | 4490    |
| BERT         | 1.00     | 1.00      | 1.00   | 1.00     | 4490    |
| DistilBERT   | 1.00     | 1.00      | 1.00   | 1.00     | 4490    |

**Class Distribution (Test Set):**
- Real News: 2142 samples  
- Fake News: 2348 samples  

---

## 🌐 Web Application (Streamlit)
A simple **web-based application** is developed using **Streamlit** to demonstrate the trained models.

### Application Features:
- Text input for news content
- Model selection via sidebar:
  - LSTM
  - BERT
  - DistilBERT
- Prediction output:
  - Classification label (Fake / Real)
  - Confidence score
  - Probability visualization (bar chart)
- Dashboard page:
  - Model comparison table
  - Confusion matrix visualization
- “How to Use” page for user guidance

---

## 📁 Project Structure
Fake-News-Detection/
│
├── app/
│ └── app.py
│
├── src/
│ ├── preprocessing.py
│ ├── infer_lstm.py
│ └── infer_transformer.py
│
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_lstm_training.ipynb
│ ├── 03_bert_training.ipynb
│ └── 04_distilbert_training.ipynb
│
├── models/ # (local only, not pushed to GitHub)
│
├── assets/
│ ├── metrics.json
│ ├── cm_lstm.png
│ ├── cm_bert.png
│ └── cm_distilbert.png
│
├── requirements.txt
├── README.md
└── .gitignore


---

## ▶️ How to Run the Application
1. Install dependencies:
```bash
pip install -r requirements.txt

2. Run the Streamlit application:
streamlit run app/app.py
Note: Trained models must be placed in the models/ directory locally before running the app.

## 🛠️ Tech Stack
- Python
- TensorFlow (LSTM)
- PyTorch (BERT & DistilBERT)
- HuggingFace Transformers
- Scikit-learn
- Pandas & NumPy
- Streamlit
- Matplotlib

## 📌 Notes
- This project is intended for academic purposes only.
- All implementations follow the UAP Machine Learning module guidelines.
- Pretrained models are used solely for educational and experimental purposes.

## 👨‍🎓 Author

Fitra Romeo Winky
Machine Learning – Final Practical Exam (UAP)