# How to Download Models?

This guide explains how to download and set up the required NLP models.

---

## LLM Files

### Llama 3.2 1B Instruct (Q4_K_M)

REQUIRED for devices under 6GB RAM

```
https://huggingface.co/lmstudio-community/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf?download=true
```

### Qwen3 4B Instruct (Q4_0)

REQUIRED for devices more than 6GB RAM

```
https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_0.gguf?download=true
```

> **Location:** Place downloaded GGUF files under `nlp_models/GGUFS`

---

## bert-emotion

Clone the BERT emotion analysis model:

```bash
cd nlp_models
git clone https://huggingface.co/boltuix/bert-emotion
```

---

## DeBERTa Illness

Clone the DeBERTa mental health detection model:

```bash
cd nlp_models
git clone https://huggingface.co/elishaw/deberta_mental
```

---

## Directory Structure

After downloading, your directory structure should look like this:

```
nlp_models/
├── GGUFS/
│   ├── Llama-3.2-1B-Instruct-Q4_K_M.gguf
│   └── Qwen3-4B-Instruct-2507-Q4_0.gguf
├── bert-emotion/
└── deberta_mental/
```
