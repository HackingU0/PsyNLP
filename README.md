# PsyNLP

A mental health text analysis tool powered by NLP and LLM technologies.

## Overview

PsyNLP is a web-based application that analyzes text (articles, journals, notes) for emotional content and potential mental health indicators. It uses transformer-based models to detect emotions and assess mental health severity.

## Features

- **Emotion Detection**: Identify emotions in text using BERT-based models
- **Mental Health Analysis**: Detect potential mental health concerns
- **Severity Scoring**: Calculate overall severity scores for analyzed text
- **LLM Enhancement**: Optional suicidal risk assessment using local LLMs
- **HTML Reports**: Generate detailed visual reports with charts
- **Web Interface**: User-friendly Flask web application
- **File Support**: Process `.txt`, `.md`, and `.docx` files

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

3. Download required models (they should be in `nlp_models/` directory, refer to [Guide](nlp_models/readme.md):
   - bert-emotion
   - deberta-illness
   - GGUF models

## Installation (for macOS)

```bash
chmod +x setup_mac.sh
./setup_mac.sh
```

## Usage

1. Start the web application:

```bash
source venv/bin/activate
python web_app.py
```

2. Open your browser and navigate to `http://localhost:7860`

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
└── web_app.py          # Main Flask application
```

## Requirements

- Python 3.12 (Does not support 3.13)
- PyTorch
- Transformers
- Flask
- spaCy
- See `requirements.txt` for full list

## License

See LICENSE file for details.

## Disclaimer

This tool is for educational and research purposes only. It should not be used as a substitute for professional mental health assessment or treatment.
