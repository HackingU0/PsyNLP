import torch
from transformers import pipeline


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

SEVERITY_SCORES = {
    "Normal": 0,
    "Stress": 2,
    "Anxiety": 4,
    "Depression": 6,
    "Bipolar": 7,
    "Personality disorder": 8,
    "Suicidal": 10,
}

def sentiment_emotion(sentence):
    pipeline_emo = pipeline("text-classification", model="nlp_models/bert-emotion")
    return pipeline_emo(sentence)


def sentiment_illness(sentence):
    pipeline_illness = pipeline(
        "text-classification", model="nlp_models/deberta_illness"
    )
    return pipeline_illness(sentence)


# Article-level Prediction
def predict_article(text: str):
    """
    Predict emotion and mental health for entire article
    Args:
        text: Full article text (combined from all sentences)
    Returns:
        dict with emotion and psychological health predictions
    """
    # Truncate text if too long
    max_length = 512
    if len(text) > max_length:
        text = text[:max_length]

    emotion_result = sentiment_emotion(text)
    illness_result = sentiment_illness(text)

    return {
        "emotion_score": emotion_result[0]["score"],
        "emotion_pred": emotion_result[0]["label"],
        "emotion_severity": SEVERITY_SCORES.get(emotion_result[0]["label"], 0),
        "psy_score": illness_result[0]["score"],
        "psy_pred": illness_result[0]["label"],
        "psy_severity": SEVERITY_SCORES.get(illness_result[0]["label"], 0),
    }


# Sentence Prediction (Deprecated, planned to be removed)
def predict_sentences(sentences: list[str]):
    results = []
    for sent in sentences:
        row = {"sentence": sent}
        emotion_result = sentiment_emotion(sent)
        illness_result = sentiment_illness(sent)
        row["emotion_score"] = emotion_result[0]["score"]
        row["emotion_pred"] = emotion_result[0]["label"]
        row["psy_score"] = illness_result[0]["score"]
        row["psy_pred"] = illness_result[0]["label"]
        results.append(row)
    return results

