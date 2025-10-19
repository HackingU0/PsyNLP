from modules.predict_nlp import predict_article
from modules.predict_llm import predict_sentences_llm
from modules.sentence_process import process_markdown


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


# Example usage
if __name__ == "__main__":
    import sys

    result = calculate_article_severity("article.md")

    print("=" * 50)
    print("文章严重性分析报告")
    print("=" * 50)
    print(f"\n总句数: {result['sentence_count']}")

    # Check for low confidence warnings
    nlp_analysis = result["nlp_analysis"]
    warnings = []
    if nlp_analysis["emotion_score"] < 0.5:
        warnings.append(f"情绪分析置信度较低: {nlp_analysis['emotion_score']:.2%}")
    if nlp_analysis["psy_score"] < 0.5:
        warnings.append(f"心理健康分类置信度较低: {nlp_analysis['psy_score']:.2%}")

    if warnings:
        print("\n⚠️  警告:")
        for warning in warnings:
            print(f"  - {warning}")

    print(f"\nNLP 整体分析:")
    print(
        f"  情绪: {nlp_analysis['emotion_pred']} (置信度: {nlp_analysis['emotion_score']:.2%})"
    )
    print(
        f"  心理健康: {nlp_analysis['psy_pred']} (置信度: {nlp_analysis['psy_score']:.2%})"
    )

    print(f"\n严重性统计:")
    print(f"  最高严重等级: {result['severity_stats']['max_severity']}/10")
    print(f"  平均严重等级: {result['severity_stats']['avg_severity']}/10")
    print(f"  高风险句子数: {result['severity_stats']['high_risk_sentences']}")
    print(f"  总体严重性评分: {result['severity_stats']['overall_severity_score']}/10")

    print(f"\n逐句分析:")
    for i, pred in enumerate(result["llm_predictions"], 1):
        print(f"\n  句子 {i}: {pred['sentence'][:60]}...")
        print(f"  分类: {pred['classification']} (严重度: {pred['severity_score']}/10)")

    print("\n" + "=" * 50)

    # Generate visualization
    print("\n生成可视化报告...")

    try:
        from modules.visualizer import generate_html_report, open_in_browser

        # Determine language from command line args
        language = "en" if "--lang=en" in sys.argv or "--english" in sys.argv else "zh"

        # Generate both languages if requested
        if "--both-lang" in sys.argv:
            print("\n生成中英文双语报告...")
            html_file_zh = generate_html_report(
                result, "mental_health_report_zh.html", language="zh"
            )
            html_file_en = generate_html_report(
                result, "mental_health_report_en.html", language="en"
            )
            html_file = html_file_zh  # Default to Chinese for opening
        else:
            html_file = generate_html_report(
                result, "mental_health_report.html", language=language
            )

        # Ask user if they want to open in browser
        if "--no-browser" not in sys.argv:
            open_in_browser(html_file)
    except Exception as e:
        print(f"无法生成可视化报告: {e}")
