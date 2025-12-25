import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

VOCAB_SIZE = 30000
MAX_LEN = 300

class LSTMInfer:
    def __init__(self, model_path: str, tokenizer_path: str):
        self.model = load_model(model_path)
        self.tokenizer = joblib.load(tokenizer_path)

    def predict(self, text: str):
        seq = self.tokenizer.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        prob_fake = float(self.model.predict(pad, verbose=0).ravel()[0])  # sigmoid
        pred = 1 if prob_fake >= 0.5 else 0
        return pred, prob_fake