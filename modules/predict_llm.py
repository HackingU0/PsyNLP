from llama_cpp import Llama
from llama_cpp.llama_types import CreateChatCompletionResponse
from typing import cast
import psutil
import re



SUICIDAL_KEYWORDS = {
    "suicide", "suicidal", "kill myself", "end my life", "end it all",
    "want to die", "wish i was dead", "can't go on", "hurt myself",
    "self-harm", "no reason to live", "everyone better off without me",
    "take my own life"
}


DEPRESSION_HINTS = {
    "hopeless", "can't get out of bed", "nothing brings me joy", "empty",
    "worthless", "pointless", "lost interest", "fatigued", "tired of life"
}


ANXIETY_HINTS = {
    "worried", "panic", "heart races", "anxious", "fear", "nervous",
    "overthinking", "can't stop thinking"
}

MARKDOWN_PATTERNS = [
    (r"```[\s\S]*?```", " "),
    (r"```[\s\S]*$", " "),
    (r"`+", " "),
    (r"\!\[[^]]*\]\([^)]*\)", " "),
    (r"\[[^]]*\]\([^)]*\)", " "),
    (r"^>+\s*", ""),
    (r"[#*_]{1,5}", ""),
    (r"-{3,}", ""),
]

_llama = None

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
    # Clean label
    label = label.strip()
    for key in SEVERITY_SCORES:
        if key.lower() in label.lower():
            return SEVERITY_SCORES[key]
    # if label not recognized
    return 0


def remove_markdown_artifacts(text: str) -> str:
    """Clean markdown / formatting symbols that can confuse LLM classification."""
    cleaned = text
    for pattern, repl in MARKDOWN_PATTERNS:
        cleaned = re.sub(pattern, repl, cleaned, flags=re.MULTILINE)
    # Remove leading bullets / numbering
    cleaned = re.sub(r"^[\s>*-]+", "", cleaned)
    cleaned = re.sub(r"^\d+\.\s+", "", cleaned)
    # Collapse excessive whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def contains_any(sentence: str, keywords: set[str]) -> bool:
    s = sentence.lower()
    for kw in keywords:
        if kw in s:
            return True
    return False

def indicates_personal_intent(sentence: str) -> bool:
    """detect first-person suicidal/self-harm intent.
    Requires pronouns/subject + intent verbs to reduce meta mentions like
    'suicidal risk warning'.
    """
    s = sentence.lower()
    pronouns = {"i", "me", "my", "myself"}
    intent_verbs = {"want", "wish", "think", "plan", "will", "might", "could", "should", "feel"}
    has_pronoun = any(f" {p} " in f" {s} " for p in pronouns)
    has_intent = any(f" {v} " in f" {s} " for v in intent_verbs)
    # direct imperative like 'kill myself' should pass even without both
    direct_patterns = [r"kill\s+myself", r"end\s+my\s+life", r"take\s+my\s+own\s+life", r"want\s+to\s+die"]
    if any(re.search(p, s) for p in direct_patterns):
        return True
    return has_pronoun and has_intent


def adjust_classification(sentence: str, raw_label: str) -> str:
    """Apply guard heuristics to reduce false positives for Suicidal.

    Logic:
    1. If raw_label == Suicidal but no strong keyword -> downgrade.
    2. If sentence is extremely short (<3 words) and model gives severe label -> downgrade to Normal.
    3. If sentence only contains formatting remnants -> Normal.
    """
    cleaned = sentence.strip()
    word_count = len(cleaned.split())
    lower_sentence = cleaned.lower()

    # If only symbols / punctuation
    if not re.search(r"[a-zA-Z]", cleaned):
        return "Normal"

    # Rule for very short sentences
    if word_count < 3 and raw_label in {"Suicidal", "Depression", "Anxiety"}:
        return "Normal"

    # Suicidal validation
    if raw_label == "Suicidal":
        if contains_any(lower_sentence, SUICIDAL_KEYWORDS) and indicates_personal_intent(lower_sentence):
            return "Suicidal"
        if contains_any(lower_sentence, DEPRESSION_HINTS):
            return "Depression"
        if contains_any(lower_sentence, ANXIETY_HINTS):
            return "Anxiety"
        if word_count >= 4:
            return "Stress"
        return "Normal"

    return raw_label

def is_thinking_model(model_path: str):
    if "thinking" in model_path.lower():
        return True

def strip_thinking_tags(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()



# Load LLM via llama-cpp-python
def load_model(
    model_path: str = "nlp_models/GGUFS/Qwen3-4B-Thinking-2507-IQ4_XS.gguf",
):
    global _llama
    if _llama is None:
        mp = model_path.lower()
        if "gemma" in mp:
            chat_format = "gemma"
        elif "qwen" in mp:
            chat_format = "chatml"
        else:
            chat_format = "llama-2"

        enable_thinking = bool("thinking" in mp or "reason" in mp)

        _llama = Llama(
            model_path=model_path,
            echo=False,
            n_ctx=16384,
            n_threads=8,
            seed=-1,
            verbose=False,
            use_mlock=True,
            chat_format=chat_format,
            enable_thinking=enable_thinking,
        )
    return _llama


memory = psutil.virtual_memory().total / (1024**3)  # Get total memory in GB
if memory < 6:  # Less than 6GB RAM, load smaller Llama3.2 1B model
    llama = load_model(model_path="nlp_models/GGUFS/gemma-3-1B-it-QAT-Q4_0.gguf")
else:
    llama = load_model(model_path="nlp_models/GGUFS/Qwen3-4B-Thinking-2507-IQ4_XS.gguf")


def classify_LLM(prompt: str):
    llama = load_model()
    temperature = 0.3 if is_thinking_model(llama.model_path) else 0.7
    system_prompt = """You are a mental health text classification system. Your task is to analyze TEXT CONTENT (not formatting) and classify it into ONE of these categories:

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
 - Ignore markdown / formatting symbols (#, *, -, >, `code`, links, bullet lists)
 - DO NOT classify as Suicidal unless explicit suicidal/self-harm language appears (e.g. "kill myself", "want to die", "end my life"). Ambiguous sadness = Depression instead.
 - Short fragments or headings without emotional content = Normal.
 - Be objective and evidence-based.
Examples:
Text: "I can't get out of bed anymore. Nothing brings me joy. What's the point?"
Classification: Depression

Text: "I'm so worried about everything. My heart races and I can't stop thinking about what could go wrong."
Classification: Anxiety

Text: "Just had a regular day at work, grabbed coffee with friends."
Classification: Normal

Text: "I don't want to be here anymore. Everyone would be better off without me."
Classification: Suicidal
"""
    if is_thinking_model(llama.model_path):
        system_prompt += "\n\nThink step-by-step: first analyze the emotional content, then determine severity, finally output ONLY the category name."
    response = llama.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": "Text: " + prompt + "\nClassification:"},
        ],
        logprobs=None,
        max_tokens=16384,
        temperature=temperature,
        top_p=0.9,
        stop=["</think>", "\n\n"],
        stream=False,
    )
    raw = str(
        cast(CreateChatCompletionResponse, response)["choices"][0]["message"]["content"]
    ).strip()
    # Strip CoT tags
    if is_thinking_model(llama.model_path):
        raw = strip_thinking_tags(raw)
    # Post-process: extract the first valid label from response
    labels = [
        "Suicidal",
        "Personality disorder",
        "Bipolar",
        "Depression",
        "Anxiety",
        "Stress",
        "Normal",
    ]
    lower = raw.lower()
    for lab in labels:
        if lab.lower() in lower:
            return lab
    # Fallback
    tokens = re.findall(r"[a-zA-Z]+(?:\s+[a-zA-Z]+)?", raw)
    for t in tokens:
        for lab in labels:
            if lab.lower() == t.strip().lower():
                return lab
    return "Normal"


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
        cleaned = remove_markdown_artifacts(sent)
        raw_label = classify_LLM(cleaned)
        adjusted_label = adjust_classification(cleaned, raw_label)
        severity = label_to_severity(adjusted_label)
        results.append(
            {
                "sentence": sent,
                "cleaned_sentence": cleaned,
                "classification": adjusted_label,
                "raw_classification": raw_label,
                "severity_score": severity,
            }
        )
    return results
