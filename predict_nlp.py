import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
import pandas as pd
import numpy
from llama_cpp import ChatCompletion
#Probably prefer Qwen 3 1.7B

# Select Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
_llama = None

def load_llama(
    model_pat:str = "nlp_models/"
)
def sentiment_emotion(sentence):
    pipeline_emo = pipeline("text-classification",model = "nlp_models/bert-emotion")
    return pipeline_emo(sentence)

def sentiment_sentiment(sentence):
    pipeline_sent = pipeline("text-classification",model = "nlp_models/bert-sentiment")
    return pipeline_sent(sentence)

def predict_sentences(sentences: list[str]):
    results = []
    for sent in sentences:
        row = {"sentence": sent}
        emotion_result = sentiment_emotion(sent)
        # Assuming emotion_result is a list with one dict containing 'label' and 'score'
        row["emotion_score"] = emotion_result[0]['score']
        row["emotion_pred"] = emotion_result[0]['label']
        results.append(row)
    return results

def to_dataframe(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(results)