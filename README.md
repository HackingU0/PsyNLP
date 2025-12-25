# PsyNLP

A Multi-Modal NLP System for Privacy-Preserving Mental Health Text Analysis

## Overview

PsyNLP is a web-based application that analyzes text (articles, journals, notes) for emotional content and potential mental health indicators. It uses transformer-based models to detect emotions and assess mental health severity.

## Features

- **Emotion Detection**: Identify emotions in text using BERT-based models
- **Mental Health Analysis**: Detect potential mental health concerns
- **Severity Scoring**: Calculate overall severity scores for analyzed text
- **LLM Enhancement**: Optional suicidal risk assessment using local LLMs
- **HTML Reports**: Generate detailed visual reports with charts
- **Web Interface**: User-friendly Streamlit web application
- **File Support**: Process `.txt`, `.md`, and `.docx` files

## Deployment (Automated)

If you have **Python 3.12** installed, you can use the automated deployment script:

```bash
python deploy.py
```

This script will handle dependencies installation and model downloading.

After deployment, run the app with:

```bash
uv run streamlit run app.py
```

## Installation (for non MacOS system)

1. Clone the repository:

```bash
git clone https://github.com/HackingU0/PsyNLP.git
cd PsyNLP
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download required models (They should be downloaded with deploy script)

## Installation (for macOS)

```bash
xcode-select --install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
git clone https://github.com/HackingU0/PsyNLP.git
brew install python@3.12
cd PsyNLP
python deploy.py
```

## Usage

1. Start the web application:

```bash
uv run streamlit run app.py
```

2. The application should open automatically in your browser. If not, navigate to the URL shown in the terminal (usually `http://localhost:8501`).

3. Upload a text file

4. View the analysis results and generated report

## Project Structure

```
PsyNLP/
├── modules/              # Core analysis modules
│   ├── file_reader.py
│   ├── sentence_process.py
│   ├── predict_nlp.py
│   ├── predict_score.py
│   ├── llm_enhancements.py
│   └── visualizer.py
├── nlp_models/          # Pre-trained models
├── templates/           # HTML templates
├── uploads/             # Uploaded files
├── reports/             # Generated reports
└── app.py              # Main Streamlit application
```

## Requirements

- Python 3.12 (Does not support 3.13)
- PyTorch
- Transformers
- Streamlit
- spaCy
- See `requirements.txt` for full list

## License

See LICENSE file for details.

## Disclaimer

This tool is for educational and research purposes only. It should not be used as a substitute for professional mental health assessment or treatment.
