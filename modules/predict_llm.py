from llama_cpp import Llama
from llama_cpp.llama_types import CreateChatCompletionResponse
from typing import cast
import psutil

_llama = None

# Severity mapping for mental health classifications
SEVERITY_SCORES = {
    "Normal": 0,
    "Stress": 2,
    "Anxiety": 4,
    "Depression": 6,
    "Bipolar": 7,
    "Personality disorder": 8,
    "Suicidal": 10,
}


def label_to_severity(label: str) -> int:
    """
    Convert LLM classification label to severity score
    Args:
        label: Classification label (Normal, Depression, etc.)
    Returns:
        Severity score (0-10)
    """
    # Clean the label (remove extra spaces, make case-insensitive)
    label = label.strip()
    for key in SEVERITY_SCORES:
        if key.lower() in label.lower():
            return SEVERITY_SCORES[key]
    # Default to 0 if label not recognized
    return 0


# Load Quantized LLM
def load_model(
    model_path: str = "nlp_models/GGUFS/Qwen3-4B-Instruct-2507-Q4_0.gguf",
):
    global _llama
    if _llama is None:
        _llama = Llama(
            model_path=model_path,
            echo=False,
            n_ctx=16384,
            n_threads=8,
            seed=-1,
            verbose=False,
        )
    return _llama


memory = psutil.virtual_memory().total / (1024**3)  # Get total memory in GB
if memory < 6:  # Less than 6GB RAM, load smaller Llama3.2 1B model
    llama = load_model(model_path="nlp_models/GGUFS/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
else:
    llama = load_model(model_path="nlp_models/GGUFS/Qwen3-4B-Instruct-2507-Q4_0.gguf")


def classify_LLM(prompt: str):
    llama = load_model()
    response = llama.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": """You are a mental health text classification system. Your task is to analyze text and classify it into ONE of these categories:

1. Normal - General conversation, no mental health concerns
2. Depression - Persistent sadness, hopelessness, loss of interest, fatigue
3. Anxiety - Excessive worry, nervousness, fear, panic symptoms
4. Stress - Feeling overwhelmed, pressure, tension
5. Bipolar - Mood swings, manic/depressive episodes
6. Personality disorder - Unstable relationships, self-image issues, impulsivity
7. Suicidal - Self-harm thoughts, suicide ideation (HIGHEST PRIORITY)

Rules:
- Output ONLY the category name
- Choose the most severe/prominent issue
- Prioritize "Suicidal" if any self-harm indicators exist
- Be objective and evidence-based""",
            },
            {
                "role": "system",
                "content": """
Classify mental health texts into these categories: Normal, Depression, Anxiety, Stress, Bipolar, Personality disorder, Suicidal.

Examples:
Text: "I can't get out of bed anymore. Nothing brings me joy. What's the point?"
Classification: Depression

Text: "I'm so worried about everything. My heart races and I can't stop thinking about what could go wrong."
Classification: Anxiety

Text: "Just had a regular day at work, grabbed coffee with friends."
Classification: Normal

Text: "I don't want to be here anymore. Everyone would be better off without me."
Classification: Suicidal""",
            },
            {"role": "user", "content": "Text: " + prompt + "\nClassification:"},
        ],
        logprobs=None,
        max_tokens=8192,
        temperature=0.7,
        top_p=0.9,
        stop=None,
        stream=False,
    )
    response = str(
        cast(CreateChatCompletionResponse, response)["choices"][0]["message"]["content"]
    ).strip()
    return response


def predict_sentences_llm(sentences: list[str]):
    """
    Predict mental health classification for each sentence using LLM
    Args:
        sentences: List of sentences to classify
    Returns:
        List of dicts with sentence, classification, and severity score
    """
    results = []
    for sent in sentences:
        classification = classify_LLM(sent)
        severity = label_to_severity(classification)
        results.append(
            {
                "sentence": sent,
                "classification": classification,
                "severity_score": severity,
            }
        )
    return results
