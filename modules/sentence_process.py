import spacy
from bs4 import BeautifulSoup
import markdown

nlp_sentence = spacy.load("en_core_web_sm")


def markdown_to_text(md_path: str) -> str:
    """Converts markdown to pure text"""
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    html = markdown.markdown(md_content)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()


def clean_text(text: str) -> str:
    """
    Remove extra line breaks, spaces, etc. and make sure each sentence is clean
    """
    import re

    # Replace consecutive line feeds and spaces with a single space
    text = re.sub(r"\s+", " ", text)
    # Remove markdown residual symbols
    text = text.replace("*", "").replace("#", "")
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """NLP Sentence Splitter"""
    doc = nlp_sentence(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def process_markdown(md_path: str) -> list[str]:
    """Processer"""
    text = markdown_to_text(md_path)
    text = clean_text(text)
    return split_sentences(text)
