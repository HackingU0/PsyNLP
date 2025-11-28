"""
Quantize prediction by using score generate by llms
"""
from modules.predict_nlp import predict_article
from modules.predict_llm import predict_sentences_llm
from modules.sentence_process import process_markdown, clean_text, split_sentences


def calculate_article_severity(md_path: str):
    """
    Calculate overall severity score for an article
    Args:
        md_path: Path to markdown file
    Returns:
        dict with detailed analysis and overall severity score
    """
    # Process markdown into sentences
    sentences = process_markdown(md_path)

    # Get article-level NLP prediction
    full_text = " ".join(sentences)
    nlp_result = predict_article(full_text)

    # Get sentence-level LLM predictions
    llm_results = predict_sentences_llm(sentences)

    # Calculate statistics
    severity_scores = []
    for result in llm_results:
        severity_scores.append(result.get("severity_score", 0))
    max_severity = max(severity_scores) if severity_scores else 0
    avg_severity = sum(severity_scores) / len(severity_scores) if severity_scores else 0
   

    # Count high-risk sentences (severity >= 6)
    high_risk_count = sum(1 for score in severity_scores if score >= 6)


    overall_severity = (
        0.3*(nlp_result.get("psy_severity",0)*nlp_result.get("psy_score",0)) + 0.7 * avg_severity
    )

    return {
        "nlp_analysis": nlp_result,
        "sentence_count": len(sentences),
        "llm_predictions": llm_results,
        "severity_stats": {
            "max_severity": max_severity,
            "avg_severity": round(avg_severity, 2),
            "high_risk_sentences": high_risk_count,
            "overall_severity_score": round(overall_severity, 2),
        },
    }


def calculate_text_severity(text: str):
    """
    Calculate overall severity score for text content directly
    Args:
        text: Text content string
    Returns:
        dict with detailed analysis and overall severity score
    """
    # Process text into sentences
    cleaned_text = clean_text(text)
    sentences = split_sentences(cleaned_text)

    # Get article-level NLP prediction
    full_text = " ".join(sentences)
    nlp_result = predict_article(full_text)

    # Get sentence-level LLM predictions
    llm_results = predict_sentences_llm(sentences)

    # Calculate statistics
    severity_scores = [result["severity_score"] for result in llm_results]
    max_severity = max(severity_scores) if severity_scores else 0
    avg_severity = sum(severity_scores) / len(severity_scores) if severity_scores else 0

    # Count high-risk sentences (severity >= 6)
    high_risk_count = sum(1 for score in severity_scores if score >= 6)

    # Overall severity calculation
    # Weighted: 50% max severity + 30% average + 20% high-risk ratio
    high_risk_ratio = high_risk_count / len(severity_scores) if severity_scores else 0
    overall_severity = (
        max_severity * 0.5 + avg_severity * 0.3 + high_risk_ratio * 10 * 0.2
    )

    return {
        "nlp_analysis": nlp_result,
        "sentence_count": len(sentences),
        "llm_predictions": llm_results,
        "severity_stats": {
            "max_severity": max_severity,
            "avg_severity": round(avg_severity, 2),
            "high_risk_sentences": high_risk_count,
            "overall_severity_score": round(overall_severity, 2),
        },
    }

