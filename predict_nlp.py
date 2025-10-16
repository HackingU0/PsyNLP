import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import numpy


# Select Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class ModelWrapper:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        self.model.eval()

    def predict(self, text: str) -> numpy.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        score = torch.softmax(outputs.logits, dim=1)
        return score.cpu().numpy()
    
def load_models(bert_path):
    return {
        "bert": ModelWrapper(bert_path)
    }

def predict_sentences(sentences: list[str], models: dict, classes: list[str]):
    results = []
    for sent in sentences:
        row = {"sentence": sent}
        for name, model in models.items():
            scores = model.predict(sent).flatten()  # shape: (num_classes,)
            # Find probability
            pred_class = classes[scores.argmax()]
            row[name + "_score"] = scores.tolist()
            row[name + "_pred"] = pred_class
        results.append(row)
    return results
def to_dataframe(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(results)