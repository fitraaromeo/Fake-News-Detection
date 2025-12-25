"""
Script untuk generate metrics dan confusion matrix dari semua model
untuk ditampilkan di Streamlit Dashboard
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support, accuracy_score

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset


def evaluate_lstm():
    """Evaluate LSTM model dan return metrics"""
    print("\n=== Evaluating LSTM ===")
    
    # Load test data
    test_df = pd.read_csv("data/processed/test.csv")
    X_test = test_df["text"].astype(str)
    y_test = test_df["label"].astype(int).values
    
    # Load model dan tokenizer
    model = load_model("models/lstm_best.h5")
    tokenizer = joblib.load("models/lstm_tokenizer.joblib")
    
    # Tokenize and pad
    MAX_LEN = 300
    test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(test_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    
    # Predict
    y_prob = model.predict(X_test_pad, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    disp.plot()
    plt.title("LSTM - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("assets/cm_lstm.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved confusion matrix to assets/cm_lstm.png")
    
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "support": int(len(y_test))
    }


def evaluate_transformer(model_dir, model_name):
    """Evaluate BERT/DistilBERT model dan return metrics"""
    print(f"\n=== Evaluating {model_name} ===")
    
    # Load test data
    test_df = pd.read_csv("data/processed/test.csv")
    test_df["text"] = test_df["text"].astype(str)
    
    # Load model dan tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    # Tokenize
    MAX_LEN = 256
    test_ds = Dataset.from_pandas(test_df)
    
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN, padding="max_length")
    
    test_tok = test_ds.map(tokenize_fn, batched=True)
    test_tok.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # Predict
    y_true = test_df["label"].values
    y_pred_list = []
    
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(test_tok), batch_size):
            batch = test_tok[i:i+batch_size]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            y_pred_list.extend(preds.cpu().numpy())
    
    y_pred = np.array(y_pred_list)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    disp.plot()
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()
    
    cm_filename = f"assets/cm_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(cm_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {cm_filename}")
    
    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "support": int(len(y_true))
    }


def main():
    # Create assets directory if not exists
    os.makedirs("assets", exist_ok=True)
    
    metrics = {}
    
    # Evaluate LSTM
    try:
        metrics["LSTM"] = evaluate_lstm()
    except Exception as e:
        print(f"Error evaluating LSTM: {e}")
        metrics["LSTM"] = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    
    # Evaluate BERT
    try:
        metrics["BERT"] = evaluate_transformer("models/bert/final", "BERT")
    except Exception as e:
        print(f"Error evaluating BERT: {e}")
        metrics["BERT"] = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    
    # Evaluate DistilBERT
    try:
        metrics["DistilBERT"] = evaluate_transformer("models/distilbert/final", "DistilBERT")
    except Exception as e:
        print(f"Error evaluating DistilBERT: {e}")
        metrics["DistilBERT"] = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
    
    # Save metrics to JSON
    metrics_path = "assets/metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Metrics saved to {metrics_path}")
    print(f"✅ Confusion matrices saved to assets/")
    
    print("\n=== Summary ===")
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {model_metrics['accuracy']:.4f}")
        print(f"  Precision: {model_metrics['precision']:.4f}")
        print(f"  Recall:    {model_metrics['recall']:.4f}")
        print(f"  F1:        {model_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
