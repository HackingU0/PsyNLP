from sentence_process import process_markdown
from predict_nlp import load_models, predict_sentences, to_dataframe
from transformers import pipeline
# Path Configuration
# MODEL_BERT_PATH = "nlp_models/model_bert"
# MODEL_SBERT_PATH = "nlp_models/model_sbert"
MODEL_EMOTION_PATH = "nlp_models/bert-emotion"
CLASSES_X = [
    "Normal",
    "Depression",
    "Anxiety",
    "Stress",
    "Bipolar",
    "Personality disorder",
    "Suicidal",
]

sentences = process_markdown("article.md")

# models = load_models(MODEL_BERT_PATH)
# results = predict_sentences(sentences, models, CLASSES)
# df = to_dataframe(results)
# print(df.head)
