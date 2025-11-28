"""
LLM Enhancement Module
Generates recommendations and professional insights
"""

from typing import Dict, cast

from llama_cpp.llama_types import CreateChatCompletionResponse

from modules.predict_llm import load_model


def generate_user_recommendations(analysis_result: Dict) -> str:
    """
    Generate mental health recommendations for general users

    Args:
        analysis_result: Result from calculate_article_severity()

    Returns:
        Personalized recommendations as string
    """
    llama = load_model()

    stats = analysis_result["severity_stats"]
    llm_predictions = analysis_result["llm_predictions"]

    # Get main issues
    classifications = [pred["classification"] for pred in llm_predictions]
    max_severity = stats["max_severity"]

    # Create context for LLM
    context = f"""Based on mental health text analysis:
- Overall severity: {stats["overall_severity_score"]}/10
- Maximum severity: {max_severity}/10
- Average severity: {stats["avg_severity"]}/10
- High-risk sentences: {stats["high_risk_sentences"]}
- Main issues detected: {", ".join(set(classifications))}
"""

    response = llama.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": """You are a compassionate mental health support assistant. Provide warm, supportive, and actionable recommendations for someone experiencing mental health challenges.

Your recommendations should:
1. Be empathetic and non-judgmental
2. Provide 3-5 specific, actionable steps
3. Include coping strategies and self-care tips
4. Mention professional help options
5. Be encouraging and hopeful
6. Keep it concise (200-300 words)

Focus on immediate support and long-term wellness.""",
            },
            {
                "role": "user",
                "content": f"{context}\n\nPlease provide personalized mental health recommendations for this person.",
            },
        ],
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
        stream=False,
    )

    return str(
        cast(CreateChatCompletionResponse, response)["choices"][0]["message"]["content"]
    ).strip()


def generate_professional_insights(analysis_result: Dict) -> Dict[str, str]:
    """
    Generate clinical insights for mental health professionals

    Args:
        analysis_result: Result from calculate_article_severity()

    Returns:
        Dictionary with professional insights
    """
    llama = load_model()

    stats = analysis_result["severity_stats"]
    llm_predictions = analysis_result["llm_predictions"]

    # Get detailed context
    classifications = [pred["classification"] for pred in llm_predictions]
    high_risk_sentences = [
        pred["sentence"] for pred in llm_predictions if pred["severity_score"] >= 6
    ]
    max_severity = stats["max_severity"]

    context = f"""Clinical Analysis Summary:
- Overall severity: {stats["overall_severity_score"]}/10
- Maximum severity: {max_severity}/10
- Average severity: {stats["avg_severity"]}/10
- High-risk indicators: {stats["high_risk_sentences"]} sentences
- Classifications: {", ".join(set(classifications))}
- High-risk content present: {"Yes" if high_risk_sentences else "No"}
"""

    # Generate DSM-5 considerations
    dsm_response = llama.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": """You are a clinical psychology expert. Provide brief DSM-5 diagnostic considerations based on text analysis.

Format:
- List 2-3 possible diagnostic categories
- Mention key symptoms/criteria observed
- Note: This is screening data, not diagnosis
- Keep it professional and concise (100-150 words)""",
            },
            {
                "role": "user",
                "content": f"{context}\n\nProvide DSM-5 diagnostic considerations.",
            },
        ],
        max_tokens=256,
        temperature=0.6,
        stream=False,
    )

    # Generate treatment recommendations
    treatment_response = llama.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": """You are a clinical psychologist. Suggest evidence-based treatment approaches.

Format:
- 2-3 recommended therapeutic modalities
- Specific interventions to consider
- Medication evaluation if warranted
- Risk management considerations
- Keep it concise (100-150 words)""",
            },
            {"role": "user", "content": f"{context}\n\nSuggest treatment approaches."},
        ],
        max_tokens=256,
        temperature=0.6,
        stream=False,
    )

    # Generate risk assessment
    risk_response = llama.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": """You are a suicide prevention specialist. Provide risk assessment guidance.

Format:
- Risk level estimation (Low/Moderate/High)
- Key risk factors identified
- Protective factors to explore
- Immediate safety recommendations
- Keep it actionable (100-150 words)""",
            },
            {"role": "user", "content": f"{context}\n\nProvide risk assessment."},
        ],
        max_tokens=256,
        temperature=0.6,
        stream=False,
    )

    return {
        "dsm5_considerations": str(
            cast(CreateChatCompletionResponse, dsm_response)["choices"][0]["message"][
                "content"
            ]
        ).strip(),
        "treatment_recommendations": str(
            cast(CreateChatCompletionResponse, treatment_response)["choices"][0][
                "message"
            ]["content"]
        ).strip(),
        "risk_assessment": str(
            cast(CreateChatCompletionResponse, risk_response)["choices"][0]["message"][
                "content"
            ]
        ).strip(),
    }


def check_suicidal_risk(analysis_result: Dict, threshold: int = 6) -> bool:
    """
    Check if there's significant suicidal risk

    Args:
        analysis_result: Result from calculate_article_severity()
        threshold: Severity threshold for concern (default 6)

    Returns:
        True if suicidal risk detected
    """
    llm_predictions = analysis_result["llm_predictions"]

    # Check for "Suicidal" classification
    for pred in llm_predictions:
        if "suicidal" in pred["classification"].lower():
            return True

    # Check for high severity scores
    stats = analysis_result["severity_stats"]
    if stats["max_severity"] >= 9:  # Very high severity
        return True

    if stats["overall_severity_score"] >= 7 and stats["high_risk_sentences"] >= 2:
        return True

    return False


if __name__ == "__main__":
    # Mock testing
    mock_result = {
        "severity_stats": {
            "overall_severity_score": 7.5,
            "max_severity": 8,
            "avg_severity": 6.2,
            "high_risk_sentences": 3,
        },
        "llm_predictions": [
            {
                "classification": "Depression",
                "severity_score": 6,
                "sentence": "I feel hopeless.",
            },
            {
                "classification": "Anxiety",
                "severity_score": 5,
                "sentence": "I'm always worried.",
            },
            {
                "classification": "Suicidal",
                "severity_score": 8,
                "sentence": "I don't want to be here.",
            },
        ],
    }

    print("=== User Recommendations ===")
    print(generate_user_recommendations(mock_result))

    print("\n=== Suicidal Risk Check ===")
    print(f"High risk detected: {check_suicidal_risk(mock_result)}")

    print("\n=== Professional Insights ===")
    insights = generate_professional_insights(mock_result)
    for key, value in insights.items():
        print(f"\n{key.upper()}:")
        print(value)
