---
tags:
  - text-classification
  - mental-health
  - deberta-v3
  - pytorch
  - transformers
  - sentiment-analysis
  - healthcare

language:
  - en

license: mit

datasets:
  - AIMH/SWMH

metrics:
  - accuracy
  - f1

pipeline_tag: text-classification
---

# DeBERTa Mental Health Classification Model

A fine-tuned DeBERTa v3 small model for detecting mental health conditions from text.

## Model Description

This model is based on `microsoft/deberta-v3-small` and has been fine-tuned to classify text into 8 mental health categories.

## Training Data

This model was trained on the following datasets:

- **SWMH (Social Media Mental Health Dataset)**: [AIMH/SWMH](https://huggingface.co/datasets/AIMH/SWMH)
- **Sentiment Analysis for Mental Health**: [Kaggle Dataset](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)

## Labels

The model can classify text into the following categories:

| ID  | Label                | Description                                         |
| --- | -------------------- | --------------------------------------------------- |
| 0   | Normal               | No mental health concerns detected                  |
| 1   | Offmychest           | General venting/sharing                             |
| 2   | Depression           | Depression-related content                          |
| 3   | Anxiety              | Anxiety-related content                             |
| 4   | Stress               | Stress-related content                              |
| 5   | Bipolar              | Bipolar disorder-related content                    |
| 6   | Personality disorder | Personality disorder-related content                |
| 7   | Suicidal             | Suicidal ideation (⚠️ requires immediate attention) |

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_path = "deberta-illness"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Example text
text = "I've been feeling down lately and can't seem to enjoy anything anymore."

# Tokenize and predict
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Get predicted label
predicted_class = torch.argmax(predictions, dim=-1).item()
confidence = predictions[0][predicted_class].item()

print(f"Predicted: {model.config.id2label[str(predicted_class)]}")
print(f"Confidence: {confidence:.2%}")
```

## Model Architecture

- **Base Model:** microsoft/deberta-v3-small
- **Hidden Size:** 768
- **Attention Heads:** 12
- **Hidden Layers:** 6
- **Max Sequence Length:** 512 tokens
- **Vocabulary Size:** 128,100

## License

Please refer to the original microsoft/deberta-v3-small license and any additional licensing terms from the fine-tuning dataset.
