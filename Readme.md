# 📰 Fake News Detection using LSTM, BERT, and DistilBERT

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)](https://streamlit.io/)

---

## 📌 Project Description

This project is developed as part of the **Final Practical Exam (Ujian Akhir Praktikum / UAP)** for the **Machine Learning** course.

**Objective:** Build a **fake news detection system** using text-based machine learning models, compare the performance between **non-pretrained models** and **pretrained (transfer learning) models**, and deploy the trained models into a **web-based application using Streamlit**.

---

## 📂 Dataset

- **Dataset Name:** Fake and Real News Dataset
- **Source:** [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

### Dataset Description

The dataset consists of two classes:

| Label | Class | Description |
|-------|-------|-------------|
| 0 | **Real News** | Factual and verified news articles |
| 1 | **Fake News** | Misleading or hoax news articles |

**Features:**
- `title`: Headline of the news article
- `text`: Full news content

**Data Split:**
- 🟦 **Training:** 80% (35,921 samples)
- 🟨 **Validation:** 10% (4,490 samples)
- 🟩 **Testing:** 10% (4,490 samples)

---

## 🧹 Data Preprocessing

The following preprocessing steps are applied to ensure clean and consistent text data:

1. ✅ Combining `title` and `text` into a single text field
2. ✅ Converting text to lowercase
3. ✅ Removing URLs and hyperlinks
4. ✅ Removing non-alphabet characters (numbers, special symbols)
5. ✅ Removing extra whitespace

**Implementation:** See [`src/preprocessing.py`](src/preprocessing.py)

---

## 🧠 Models Implemented

This project implements **three machine learning models** as required by the UAP guidelines:

### 1️⃣ LSTM (Non-Pretrained Model)

- **Architecture:** Bidirectional LSTM with Embedding Layer
- **Training:** Built from scratch using TensorFlow/Keras
- **Purpose:** Serves as a baseline model for text classification
- **Features:**
  - Custom vocabulary tokenizer
  - 30,000 vocabulary size
  - 128-dimensional embeddings
  - Sequence length: 300 tokens

### 2️⃣ BERT (Pretrained Model – Transfer Learning)

- **Base Model:** `bert-base-uncased` from HuggingFace
- **Fine-tuning:** 2 epochs with learning rate 2e-5
- **Features:**
  - 110M parameters
  - Max sequence length: 256 tokens
  - Binary classification head

### 3️⃣ DistilBERT (Pretrained Model – Transfer Learning)

- **Base Model:** `distilbert-base-uncased` from HuggingFace
- **Fine-tuning:** 2 epochs with learning rate 2e-5
- **Features:**
  - 66M parameters (40% smaller than BERT)
  - Faster inference time
  - Max sequence length: 256 tokens

---

## 📊 Model Evaluation

All models are evaluated using the **test dataset** (4,490 samples) with standard classification metrics.

### 📈 Evaluation Results (Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | Support |
|-------|----------|-----------|--------|----------|---------|
| **LSTM** | 99.91% | 99.87% | 99.96% | 99.91% | 4,490 |
| **BERT** | 99.96% | 99.96% | 99.96% | 99.96% | 4,490 |
| **DistilBERT** | 99.89% | 99.91% | 99.87% | 99.89% | 4,490 |

**Test Set Distribution:**
- Real News: 2,142 samples (47.7%)
- Fake News: 2,348 samples (52.3%)

### 🏆 Key Findings

- ✅ All models achieved **>99.8% accuracy** on the test set
- ✅ **BERT** achieved the highest overall performance (99.96%)
- ✅ **DistilBERT** provides comparable performance with faster inference
- ✅ **LSTM** baseline model performs surprisingly well (99.91%)

**Confusion matrices and detailed metrics are available in the Streamlit dashboard.**

---

## 🌐 Web Application (Streamlit)

A fully interactive **web-based application** is developed using **Streamlit** to demonstrate the trained models in real-time.

### ✨ Application Features

#### 📊 Dashboard Page
- Model performance comparison table
- Visual confusion matrices for all models
- Summary statistics and metrics

#### 🔮 Predict Page
- **Text Input:** Paste news article (title + content)
- **Model Selection:** Choose between LSTM, BERT, or DistilBERT
- **Preprocessing Toggle:** Enable/disable text cleaning
- **LSTM Threshold Slider:** Adjust classification threshold
- **Real-time Prediction:**
  - Classification label (Real/Fake)
  - Confidence score with progress bar
  - Probability distribution visualization

#### 📘 How to Use Page
- Step-by-step guide for running the application
- Model file requirements
- Usage instructions

---

## 📁 Project Structure

```
Fake-News-Detection/
│
├── app/
│   └── app.py                      # Streamlit web application
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Text preprocessing utilities
│   ├── infer_lstm.py               # LSTM inference class
│   └── infer_transformer.py        # BERT/DistilBERT inference class
│
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory Data Analysis
│   ├── 02_lstm_training.ipynb      # LSTM model training
│   ├── 03_bert_training.ipynb      # BERT model training
│   └── 04_distilbert_training.ipynb # DistilBERT model training
│
├── data/
│   ├── raw/                        # Original dataset (Fake.csv, True.csv)
│   └── processed/                  # Preprocessed splits (train/val/test)
│
├── models/                         # ⚠️ Local only (not pushed to GitHub)
│   ├── lstm_best.h5                # Trained LSTM model
│   ├── lstm_tokenizer.joblib       # LSTM tokenizer
│   ├── bert/final/                 # BERT model & tokenizer
│   └── distilbert/final/           # DistilBERT model & tokenizer
│
├── assets/
│   ├── metrics.json                # Model evaluation metrics
│   ├── cm_lstm.png                 # LSTM confusion matrix
│   ├── cm_bert.png                 # BERT confusion matrix
│   └── cm_distilbert.png           # DistilBERT confusion matrix
│
├── generate_metrics.py             # Script to generate metrics & confusion matrices
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore rules
```


---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster training/inference

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Prepare Models

**Option A:** Train models from scratch using the provided notebooks:
1. Run `01_eda.ipynb` for data exploration
2. Run `02_lstm_training.ipynb` to train LSTM
3. Run `03_bert_training.ipynb` to train BERT
4. Run `04_distilbert_training.ipynb` to train DistilBERT

**Option B:** Download pre-trained models (if available) and place them in the `models/` directory.

### Step 4: Generate Metrics & Confusion Matrices

```bash
python generate_metrics.py
```

This will create `assets/metrics.json` and confusion matrix images.

### Step 5: Run the Streamlit Application

```bash
streamlit run app/app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.10 |
| **Deep Learning** | TensorFlow 2.x, PyTorch 2.x |
| **NLP** | HuggingFace Transformers |
| **Data Processing** | Pandas, NumPy, Scikit-learn |
| **Visualization** | Matplotlib, Seaborn |
| **Web Framework** | Streamlit |
| **Model Serialization** | Joblib, HuggingFace SafeTensors |

---

## 📝 Usage Example

### Command Line Inference

```python
from src.infer_lstm import LSTMInfer
from src.preprocessing import clean_text

# Load model
lstm = LSTMInfer("models/lstm_best.h5", "models/lstm_tokenizer.joblib")

# Predict
text = "Breaking news: Scientists discover new planet!"
cleaned = clean_text(text)
pred, prob = lstm.predict(cleaned)

print(f"Prediction: {'FAKE' if pred == 1 else 'REAL'}")
print(f"Confidence: {prob:.4f}")
```

---

## 📌 Notes

- ⚠️ This project is intended for **academic purposes only**.
- ✅ All implementations follow the **UAP Machine Learning module guidelines**.
- 🎓 Pretrained models are used solely for **educational and experimental purposes**.
- 🔒 Trained model files are **not included in the repository** due to size constraints.
- 📊 Dataset is publicly available on Kaggle under appropriate license.

---

## 🤝 Contributing

This is an academic project. However, suggestions and feedback are welcome!

---

## 📄 License

This project is developed for educational purposes as part of the Machine Learning course requirements.

---

## 👨‍🎓 Author

**Fitra Romeo Winky**  
Machine Learning – Final Practical Exam (UAP)  
Semester 7, 2025

---

## 🙏 Acknowledgments

- Kaggle for providing the Fake and Real News Dataset
- HuggingFace for pretrained transformer models
- TensorFlow and PyTorch teams for excellent deep learning frameworks
- Streamlit for the intuitive web app framework

---

**⭐ If you find this project helpful, please consider giving it a star!**