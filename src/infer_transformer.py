import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TransformerInfer:
    def __init__(self, model_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy().ravel()  # [real, fake]
        pred = int(probs.argmax())
        prob_pred = float(probs[pred])
        return pred, probs, prob_pred