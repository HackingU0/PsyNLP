import os
import re
from datetime import datetime
from typing import Dict, List
from jinja2 import Environment, FileSystemLoader, select_autoescape


# Import enhancements

try:
    from modules.llm_enhancements import (
        generate_user_recommendations,
        generate_professional_insights,
    )

    ENHANCEMENTS_AVAILABLE = True
except ImportError:
    ENHANCEMENTS_AVAILABLE = False
    print("Warning: LLM enhancements not available")


def markdown_to_html(text: str) -> str:
    """
    Convert simple Markdown formatting to HTML
    Supports: **bold**, *italic*, bullet points, numbered lists

    Args:
        text: Text with Markdown formatting

    Returns:
        HTML formatted text
    """
    if not text:
        return text

    # Convert **bold**
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)

    # Convert *italic*
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)

    # Convert __bold__
    text = re.sub(r"__([^_]+)__", r"<strong>\1</strong>", text)

    # Convert _italic_
    text = re.sub(r"_([^_]+)_", r"<em>\1</em>", text)

    # Process lists
    lines = text.split("\n")
    result_lines = []
    current_list_type = None  # Track: 'ul', 'ol', or None
    i = 0

    # check for next list item
    def has_next_numbered_item(start_idx, lines, max_lookahead=10):
        for j in range(start_idx, min(start_idx + max_lookahead, len(lines))):
            if re.match(r"^\d+\.\s+", lines[j].strip()):
                return True
        return False

    def has_next_bullet_item(start_idx, lines, max_lookahead=10):
        for j in range(start_idx, min(start_idx + max_lookahead, len(lines))):
            if re.match(r"^[-*]\s+", lines[j].strip()):
                return True
        return False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check for numbered list
        if re.match(r"^\d+\.\s+", stripped):
            # Close previous list if it's a different type
            if current_list_type == "ul":
                result_lines.append("</ul>")
                current_list_type = None

            # Start ordered list
            if current_list_type != "ol":
                result_lines.append("<ol>")
                current_list_type = "ol"

            # Extract content after the number
            item_content = re.sub(r"^\d+\.\s+", "", stripped)

            # multiline content
            multiline_content = [item_content]
            j = i + 1

            # Collect lines until we hit the next numbered item or end
            while j < len(lines):
                next_line = lines[j].strip()
                if re.match(r"^\d+\.\s+", next_line):
                    break
                if next_line:
                    multiline_content.append(next_line)
                j += 1

            full_content = "<br>".join(multiline_content)
            result_lines.append(f"  <li>{full_content}</li>")

            
            i = j - 1
        elif re.match(r"^[-*]\s+", stripped):
            if current_list_type == "ol":
                result_lines.append("</ol>")
                current_list_type = None

            
            if current_list_type != "ul":
                result_lines.append("<ul>")
                current_list_type = "ul"

            # Extract content after the bullet
            item_content = re.sub(r"^[-*]\s+", "", stripped)

            # multiline content
            multiline_content = [item_content]
            j = i + 1

            # Collect lines until we hit the next bullet item or end
            while j < len(lines):
                next_line = lines[j].strip()
                if re.match(r"^[-*]\s+", next_line):
                    break
                if next_line:
                    multiline_content.append(next_line)
                j += 1

            full_content = "<br>".join(multiline_content)
            result_lines.append(f"  <li>{full_content}</li>")

            i = j - 1

        # Empty line
        elif not stripped:
            if not current_list_type:
                result_lines.append(line)

        # Regular line with content (not a list item)
        else:
            if current_list_type == "ol" and not has_next_numbered_item(i + 1, lines):
                result_lines.append("</ol>")
                current_list_type = None
            elif current_list_type == "ul" and not has_next_bullet_item(i + 1, lines):
                result_lines.append("</ul>")
                current_list_type = None

            if not current_list_type:
                result_lines.append(line)

        i += 1

    if current_list_type == "ul":
        result_lines.append("</ul>")
    elif current_list_type == "ol":
        result_lines.append("</ol>")

    text = "\n".join(result_lines)

    
    parts = re.split(r"(</?(?:ul|ol|li)>)", text)
    result_parts = []
    in_html_list = False

    for part in parts:
        if part in ["<ul>", "<ol>"]:
            in_html_list = True
            result_parts.append(part)
        elif part in ["</ul>", "</ol>"]:
            in_html_list = False
            result_parts.append(part)
        elif "<li>" in part or "</li>" in part:
            result_parts.append(part)
        else:
            if not in_html_list:
                # Replace double newlines with paragraph breaks
                part = re.sub(r"\n\n+", "</p><p>", part)
                # Replace single newlines with <br>
                part = re.sub(r"\n", "<br>", part)
            result_parts.append(part)

    text = "".join(result_parts)

    return text


# Color mapping for severity levels
SEVERITY_COLORS = {
    0: "#28a745",  # Green - Normal
    2: "#90ee90",  # Light Green - Stress
    4: "#ffc107",  # Yellow - Anxiety
    6: "#ff9800",  # Orange - Depression
    7: "#ff6b35",  # Dark Orange - Bipolar
    8: "#e74c3c",  # Red - Personality disorder
    10: "#c0392b",  # Dark Red - Suicidal
}


def get_severity_color(severity: int) -> str:
    """Get color for severity score"""
    for level in sorted(SEVERITY_COLORS.keys(), reverse=True):
        if severity >= level:
            return SEVERITY_COLORS[level]
    return SEVERITY_COLORS[0]


def check_low_confidence(analysis_result: Dict, threshold: float = 0.5) -> List[Dict]:
    """
    Check for low confidence scores in NLP analysis
    Args:
        analysis_result: Result from calculate_article_severity()
        threshold: Confidence threshold (default 0.5 = 50%)
    Returns:
        List of low confidence warnings
    """
    warnings = []
    nlp_analysis = analysis_result["nlp_analysis"]

    if nlp_analysis["emotion_score"] < threshold:
        warnings.append(
            {
                "metric": "Emotion Analysis",
                "label": nlp_analysis["emotion_pred"],
                "confidence": nlp_analysis["emotion_score"],
            }
        )

    if nlp_analysis["psy_score"] < threshold:
        warnings.append(
            {
                "metric": "Mental Health Classification",
                "label": nlp_analysis["psy_pred"],
                "confidence": nlp_analysis["psy_score"],
            }
        )

    return warnings


def generate_html_report(
    analysis_result: Dict,
    output_path: str = "report.html",
    language: str = "en",
    mode: str = "user",
):
    """
    Generate an HTML report with mode-specific content using Jinja2 templates
    Args:
        analysis_result: Result from calculate_article_severity()
        output_path: Path to save HTML file
        language: Report language (always 'en' now)
        mode: 'user' or 'professional'
    """

    low_confidence_warnings = check_low_confidence(analysis_result)

    nlp_analysis = analysis_result["nlp_analysis"]
    stats = analysis_result["severity_stats"]
    llm_predictions = analysis_result["llm_predictions"]
    sentence_count = analysis_result["sentence_count"]

    # Calculate severity distribution
    severity_distribution = {}
    for pred in llm_predictions:
        label = pred["classification"]
        severity_distribution[label] = severity_distribution.get(label, 0) + 1

 
    distribution_list = []
    for label, count in sorted(
        severity_distribution.items(), key=lambda x: x[1], reverse=True
    ):
        # Find severity score for this label
        severity = next(
            (pred["severity_score"] for pred in llm_predictions if pred["classification"] == label),
            0
        )
        distribution_list.append({
            "name": label,
            "count": count,
            "color": get_severity_color(severity)
        })

 
    predictions_with_colors = []
    for pred in llm_predictions:
        pred_copy = pred.copy()
        pred_copy["color"] = get_severity_color(pred["severity_score"])
        predictions_with_colors.append(pred_copy)

    # Generate mode-specific content
    recommendations_html = None
    professional_insights = None
    
    if ENHANCEMENTS_AVAILABLE:
        if mode == "user":
            recommendations = generate_user_recommendations(analysis_result)  # type: ignore
            recommendations_html = markdown_to_html(recommendations)
        else:  # professional mode
            insights = generate_professional_insights(analysis_result)  # type: ignore
            professional_insights = {
                "dsm5_html": markdown_to_html(insights["dsm5_considerations"]),
                "treatment_html": markdown_to_html(insights["treatment_recommendations"]),
                "risk_html": markdown_to_html(insights["risk_assessment"])
            }


    disclaimer = (
        "This report is for reference only and cannot replace professional medical diagnosis. Please seek professional help if you have serious mental health issues."
        if mode == "user"
        else "This analysis is a screening tool for clinical use. It should be used in conjunction with clinical interview and professional judgment."
    )

    template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(['html', 'xml'])
    )

    env.filters['get_severity_color'] = get_severity_color
    

    template = env.get_template("report.html.j2")
    

    html_content = template.render(
        mode=mode,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        sentence_count=sentence_count,
        low_confidence_warnings=low_confidence_warnings,
        stats=stats,
        nlp_analysis=nlp_analysis,
        recommendations_html=recommendations_html,
        professional_insights=professional_insights,
        severity_distribution=distribution_list,
        llm_predictions=predictions_with_colors,
        disclaimer=disclaimer,
        get_severity_color=get_severity_color
    )


    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML report generated: {output_path}")
    return output_path


def generate_pdf_report(
    analysis_result: Dict,
    output_path: str = "report.pdf",
    language: str = "en",
    mode: str = "user",
):
    """
    Generate PDF report from HTML
    """
    try:
        from weasyprint import HTML

        # First generate HTML
        html_path = output_path.replace(".pdf", ".html")
        generate_html_report(analysis_result, html_path, language, mode)

        # Convert to PDF
        HTML(html_path).write_pdf(output_path)
        print(f"PDF report generated: {output_path}")

        return output_path
    except ImportError:
        print("Unable to generate PDF: please install weasyprint")
        print("   Installation: pip install weasyprint")
        print("   Or use browser to print HTML to PDF")
        return None


def open_in_browser(html_path: str):
    """Open HTML report in default browser"""
    import webbrowser
    import os

    abs_path = os.path.abspath(html_path)
    webbrowser.open(f"file://{abs_path}")
    print(f"Report opened in browser")



if __name__ == "__main__":
    from modules.predict_score import calculate_article_severity

    result = calculate_article_severity("article.md")

    print("\nGenerating User Report")
    html_user = generate_html_report(result, "report_user.html", mode="user")


    print("\nGenerating Professional Report")
    html_prof = generate_html_report(
        result, "report_professional.html", mode="professional"
    )


    open_in_browser(html_user)
