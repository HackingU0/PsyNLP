"""
Enhanced Visualizer for User and Professional Modes
English-only version with mode-specific features
"""

import os
import re
from datetime import datetime
from typing import Dict, List


# Import enhancements

try:
    from modules.llm_enhancements import (
        generate_user_recommendations,
        generate_professional_insights,
        check_suicidal_risk,
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

    # Convert **bold** to <strong>bold</strong>
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)

    # Convert *italic* to <em>italic</em>
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)

    # Convert __bold__ to <strong>bold</strong>
    text = re.sub(r"__([^_]+)__", r"<strong>\1</strong>", text)

    # Convert _italic_ to <em>italic</em>
    text = re.sub(r"_([^_]+)_", r"<em>\1</em>", text)

    # Process lists line by line with lookahead for multiline items
    lines = text.split("\n")
    result_lines = []
    current_list_type = None  # Track: 'ul', 'ol', or None
    i = 0

    # Helper function to check if next numbered item exists within N lines
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

        # Check for numbered list (1. item, 2. item, etc.)
        if re.match(r"^\d+\.\s+", stripped):
            # Close previous list if it's a different type
            if current_list_type == "ul":
                result_lines.append("</ul>")
                current_list_type = None

            # Start ordered list if not already in one
            if current_list_type != "ol":
                result_lines.append("<ol>")
                current_list_type = "ol"

            # Extract content after the number
            item_content = re.sub(r"^\d+\.\s+", "", stripped)

            # Look ahead to collect multiline content for this item
            multiline_content = [item_content]
            j = i + 1

            # Collect lines until we hit the next numbered item or end
            while j < len(lines):
                next_line = lines[j].strip()
                # Stop if we hit another numbered item
                if re.match(r"^\d+\.\s+", next_line):
                    break
                # Add non-empty lines to this item
                if next_line:
                    multiline_content.append(next_line)
                j += 1

            # Combine all content for this list item
            full_content = "<br>".join(multiline_content)
            result_lines.append(f"  <li>{full_content}</li>")

            # Skip the lines we've already processed
            i = j - 1

        # Check for bullet list (- item or * item)
        elif re.match(r"^[-*]\s+", stripped):
            # Close previous list if it's a different type
            if current_list_type == "ol":
                result_lines.append("</ol>")
                current_list_type = None

            # Start unordered list if not already in one
            if current_list_type != "ul":
                result_lines.append("<ul>")
                current_list_type = "ul"

            # Extract content after the bullet
            item_content = re.sub(r"^[-*]\s+", "", stripped)

            # Look ahead to collect multiline content for this item
            multiline_content = [item_content]
            j = i + 1

            # Collect lines until we hit the next bullet item or end
            while j < len(lines):
                next_line = lines[j].strip()
                # Stop if we hit another bullet item
                if re.match(r"^[-*]\s+", next_line):
                    break
                # Add non-empty lines to this item
                if next_line:
                    multiline_content.append(next_line)
                j += 1

            # Combine all content for this list item
            full_content = "<br>".join(multiline_content)
            result_lines.append(f"  <li>{full_content}</li>")

            # Skip the lines we've already processed
            i = j - 1

        # Empty line
        elif not stripped:
            # Skip empty lines if we're in a list
            if not current_list_type:
                result_lines.append(line)

        # Regular line with content (not a list item)
        else:
            # Only close the list if there's no upcoming list item of the same type
            if current_list_type == "ol" and not has_next_numbered_item(i + 1, lines):
                result_lines.append("</ol>")
                current_list_type = None
            elif current_list_type == "ul" and not has_next_bullet_item(i + 1, lines):
                result_lines.append("</ul>")
                current_list_type = None

            # If we closed the list, add this line
            if not current_list_type:
                result_lines.append(line)

        i += 1

    # Close any remaining open list
    if current_list_type == "ul":
        result_lines.append("</ul>")
    elif current_list_type == "ol":
        result_lines.append("</ol>")

    text = "\n".join(result_lines)

    # Convert newlines to <br> but not inside lists
    # Split by HTML tags to avoid adding <br> inside them
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
    Generate an HTML report with mode-specific content
    Args:
        analysis_result: Result from calculate_article_severity()
        output_path: Path to save HTML file
        language: Report language (always 'en' now)
        mode: 'user' or 'professional'
    """

    # Check for low confidence warnings
    low_confidence_warnings = check_low_confidence(analysis_result)

    # Extract data
    nlp_analysis = analysis_result["nlp_analysis"]
    stats = analysis_result["severity_stats"]
    llm_predictions = analysis_result["llm_predictions"]
    sentence_count = analysis_result["sentence_count"]

    # Calculate severity distribution
    severity_distribution = {}
    for pred in llm_predictions:
        label = pred["classification"]
        severity_distribution[label] = severity_distribution.get(label, 0) + 1

    # Generate mode-specific content
    mode_content = ""
    if ENHANCEMENTS_AVAILABLE:
        if mode == "user":
            recommendations = generate_user_recommendations(analysis_result)  # type: ignore
            # Convert Markdown to HTML
            recommendations_html = markdown_to_html(recommendations)
            mode_content = f"""
        <!-- User Recommendations -->
        <div class="section">
            <h2 class="section-title">💡 Personalized Recommendations</h2>
            <div class="recommendation-box">
                <div style="line-height: 1.8;">{recommendations_html}</div>
            </div>
            
            <div class="resources-box">
                <h3>🆘 Crisis Resources</h3>
                <ul>
                    <li><strong>988 Suicide & Crisis Lifeline:</strong> Call or text 988</li>
                    <li><strong>Crisis Text Line:</strong> Text 'HELLO' to 741741</li>
                    <li><strong>SAMHSA National Helpline:</strong> 1-800-662-4357</li>
                    <li><strong>International:</strong> <a href="https://findahelpline.com" target="_blank">findahelpline.com</a></li>
                </ul>
            </div>
        </div>
"""
        else:  # professional mode
            insights = generate_professional_insights(analysis_result)  # type: ignore
            # Convert Markdown to HTML for each section
            dsm5_html = markdown_to_html(insights["dsm5_considerations"])
            treatment_html = markdown_to_html(insights["treatment_recommendations"])
            risk_html = markdown_to_html(insights["risk_assessment"])

            mode_content = f"""
        <!-- Professional Insights -->
        <div class="section">
            <h2 class="section-title">🏥 Clinical Assessment</h2>
            
            <div class="professional-box">
                <h3>📋 DSM-5 Diagnostic Considerations</h3>
                <div style="line-height: 1.8;">{dsm5_html}</div>
            </div>
            
            <div class="professional-box">
                <h3>💊 Treatment Recommendations</h3>
                <div style="line-height: 1.8;">{treatment_html}</div>
            </div>
            
            <div class="professional-box risk-assessment">
                <h3>⚠️ Risk Assessment</h3>
                <div style="line-height: 1.8;">{risk_html}</div>
            </div>
            
            <div class="clinical-note">
                <strong>Clinical Note:</strong> This analysis is a screening tool only. 
                It should be used in conjunction with clinical interview, patient history, 
                and professional judgment. Not a substitute for comprehensive psychiatric evaluation.
            </div>
        </div>
"""

    # HTML template
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mental Health Analysis Report - {mode.title()} Mode</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f5f7fa;
            padding: 20px;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
        }}
        
        .header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
        }}
        
        .header .meta {{
            opacity: 0.9;
            font-size: 14px;
        }}
        
        .mode-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            background: rgba(255,255,255,0.2);
            font-size: 12px;
            margin-top: 10px;
        }}
        
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            padding: 30px;
            background-color: #f8f9fa;
        }}
        
        .card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            border-left: 4px solid;
        }}
        
        .card.normal {{ border-left-color: #28a745; }}
        .card.warning {{ border-left-color: #ffc107; }}
        .card.danger {{ border-left-color: #dc3545; }}
        
        .card-title {{
            font-size: 14px;
            color: #6c757d;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .card-value {{
            font-size: 32px;
            font-weight: bold;
            color: #2c3e50;
        }}
        
        .card-subtitle {{
            font-size: 14px;
            color: #95a5a6;
            margin-top: 5px;
        }}
        
        .section {{
            padding: 30px;
            border-bottom: 1px solid #e9ecef;
        }}
        
        .section:last-child {{
            border-bottom: none;
        }}
        
        .section-title {{
            font-size: 22px;
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        
        .warning-box {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 20px;
            margin: 20px 0;
            border-radius: 6px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        
        .warning-box h3 {{
            color: #856404;
            margin-bottom: 15px;
            font-size: 18px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .warning-item {{
            background: white;
            padding: 12px;
            margin: 8px 0;
            border-radius: 4px;
            border-left: 3px solid #ff9800;
        }}
        
        .recommendation-box {{
            background: #e3f2fd;
            border-left: 4px solid #2196F3;
            padding: 20px;
            margin: 20px 0;
            border-radius: 6px;
            line-height: 1.8;
        }}
        
        .resources-box {{
            background: #fff8e1;
            border-left: 4px solid #ff9800;
            padding: 20px;
            margin: 20px 0;
            border-radius: 6px;
        }}
        
        .resources-box h3 {{
            color: #f57c00;
            margin-bottom: 15px;
        }}
        
        .resources-box ul {{
            list-style: none;
            padding-left: 0;
        }}
        
        .resources-box li {{
            padding: 8px 0;
            border-bottom: 1px solid #ffe0b2;
        }}
        
        .resources-box li:last-child {{
            border-bottom: none;
        }}
        
        .professional-box {{
            background: #f5f5f5;
            border-left: 4px solid #607d8b;
            padding: 20px;
            margin: 15px 0;
            border-radius: 6px;
        }}
        
        .professional-box h3 {{
            color: #455a64;
            margin-bottom: 12px;
            font-size: 16px;
        }}
        
        .professional-box.risk-assessment {{
            background: #ffebee;
            border-left-color: #d32f2f;
        }}
        
        .professional-box.risk-assessment h3 {{
            color: #c62828;
        }}
        
        .clinical-note {{
            background: #e8f5e9;
            border: 1px solid #66bb6a;
            padding: 15px;
            margin: 20px 0;
            border-radius: 6px;
            font-size: 13px;
            color: #2e7d32;
        }}
        
        .nlp-results {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
        }}
        
        .nlp-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
        }}
        
        .nlp-item strong {{
            display: block;
            color: #495057;
            margin-bottom: 8px;
        }}
        
        .badge {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 14px;
            font-weight: 600;
        }}
        
        .distribution-chart {{
            display: flex;
            gap: 15px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        
        .distribution-item {{
            flex: 1;
            min-width: 120px;
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 6px;
        }}
        
        .distribution-count {{
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .distribution-label {{
            font-size: 13px;
            color: #6c757d;
        }}
        
        .sentence-list {{
            margin-top: 20px;
        }}
        
        .sentence-item {{
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid;
            transition: transform 0.2s;
        }}
        
        .sentence-item:hover {{
            transform: translateX(5px);
        }}
        
        .sentence-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        
        .sentence-number {{
            font-weight: bold;
            color: #667eea;
        }}
        
        .sentence-text {{
            color: #2c3e50;
            line-height: 1.8;
            margin-bottom: 10px;
        }}
        
        .sentence-meta {{
            display: flex;
            gap: 15px;
            font-size: 13px;
        }}
        
        .severity-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 5px;
        }}
        
        .risk-level {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
        }}
        
        .risk-low {{ background: #d4edda; color: #155724; }}
        .risk-medium {{ background: #fff3cd; color: #856404; }}
        .risk-high {{ background: #f8d7da; color: #721c24; }}
        
        .footer {{
            padding: 20px 30px;
            background: #f8f9fa;
            text-align: center;
            color: #6c757d;
            font-size: 13px;
        }}
        
        @media print {{
            body {{ background: white; padding: 0; }}
            .container {{ box-shadow: none; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🧠 Mental Health Text Analysis Report</h1>
            <div class="mode-badge">{'👤 USER MODE' if mode == 'user' else '🏥 PROFESSIONAL MODE'}</div>
            <div class="meta">
                Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | 
                Sentences Analyzed: {sentence_count} | 
                System Version: PsyNLP v1.0
            </div>
        </div>
        
        <!-- Low Confidence Warning (if any) -->
"""

    # Add warning section if there are low confidence results
    if low_confidence_warnings:
        html_content += """
        <div class="section">
            <div class="warning-box">
                <h3>
                    <span style="font-size: 24px;">⚠️</span>
                    Warning: Low Confidence
                </h3>
                <p>The following metrics have confidence below 50%, results may not be accurate:</p>
"""
        for warning in low_confidence_warnings:
            html_content += f"""
                <div class="warning-item">
                    <strong>{warning['metric']}:</strong> {warning['label']} 
                    (Confidence: {warning['confidence']:.2%})
                </div>
"""
        html_content += """
            </div>
        </div>
"""

    html_content += f"""
        
        <!-- Summary Cards -->
        <div class="summary-cards">
            <div class="card {'danger' if stats['overall_severity_score'] >= 6 else 'warning' if stats['overall_severity_score'] >= 3 else 'normal'}">
                <div class="card-title">Overall Severity Score</div>
                <div class="card-value">{stats['overall_severity_score']}/10</div>
                <div class="card-subtitle">Comprehensive Assessment</div>
            </div>
            
            <div class="card {'danger' if stats['max_severity'] >= 6 else 'warning' if stats['max_severity'] >= 3 else 'normal'}">
                <div class="card-title">Maximum Risk Level</div>
                <div class="card-value">{stats['max_severity']}/10</div>
                <div class="card-subtitle">Most Severe Sentence Score</div>
            </div>
            
            <div class="card {'danger' if stats['high_risk_sentences'] > 0 else 'normal'}">
                <div class="card-title">High-Risk Sentences</div>
                <div class="card-value">{stats['high_risk_sentences']}</div>
                <div class="card-subtitle">Sentences with severity ≥ 6</div>
            </div>
            
            <div class="card normal">
                <div class="card-title">Average Severity</div>
                <div class="card-value">{stats['avg_severity']}/10</div>
                <div class="card-subtitle">Document Average Level</div>
            </div>
        </div>
        
        <!-- NLP Analysis Section -->
        <div class="section">
            <h2 class="section-title">📊 NLP Overall Analysis</h2>
            <div class="nlp-results">
                <div class="nlp-item">
                    <strong>Emotion Analysis</strong>
                    <span class="badge" style="background-color: {get_severity_color(int(nlp_analysis['emotion_score'] * 10))}; color: white;">
                        {nlp_analysis['emotion_pred']}
                    </span>
                    <div style="margin-top: 10px;">
                        Confidence: {nlp_analysis['emotion_score']:.2%}
                        {' ⚠️' if nlp_analysis['emotion_score'] < 0.5 else ''}
                    </div>
                </div>
                
                <div class="nlp-item">
                    <strong>Mental Health Classification</strong>
                    <span class="badge" style="background-color: {get_severity_color(int(nlp_analysis['psy_score'] * 10))}; color: white;">
                        {nlp_analysis['psy_pred']}
                    </span>
                    <div style="margin-top: 10px;">
                        Confidence: {nlp_analysis['psy_score']:.2%}
                        {' ⚠️' if nlp_analysis['psy_score'] < 0.5 else ''}
                    </div>
                </div>
            </div>
        </div>
        
        {mode_content}
        
        <!-- Severity Distribution -->
        <div class="section">
            <h2 class="section-title">📈 Classification Distribution</h2>
            <div class="distribution-chart">
"""

    # Add distribution items
    for label, count in sorted(
        severity_distribution.items(), key=lambda x: x[1], reverse=True
    ):
        html_content += f"""
                <div class="distribution-item">
                    <div class="distribution-count" style="color: {get_severity_color(next((pred['severity_score'] for pred in llm_predictions if pred['classification'] == label), 0))};">
                        {count}
                    </div>
                    <div class="distribution-label">{label}</div>
                </div>
"""

    html_content += """
            </div>
        </div>
        
        <!-- Sentence Analysis -->
        <div class="section">
            <h2 class="section-title">📝 Detailed Sentence Analysis</h2>
            <div class="sentence-list">
"""

    # Add sentence items
    for i, pred in enumerate(llm_predictions, 1):
        severity = pred["severity_score"]
        color = get_severity_color(severity)
        html_content += f"""
                <div class="sentence-item" style="border-left-color: {color};">
                    <div class="sentence-header">
                        <span class="sentence-number">Sentence #{i}</span>
                        <span class="risk-level {'risk-high' if severity >= 6 else 'risk-medium' if severity >= 3 else 'risk-low'}">
                            Severity: {severity}/10
                        </span>
                    </div>
                    <div class="sentence-text">"{pred['sentence']}"</div>
                    <div class="sentence-meta">
                        <span>
                            <span class="severity-indicator" style="background-color: {color};"></span>
                            Classification: <strong>{pred['classification']}</strong>
                        </span>
                    </div>
                </div>
"""

    disclaimer = (
        "This report is for reference only and cannot replace professional medical diagnosis. Please seek professional help if you have serious mental health issues."
        if mode == "user"
        else "This analysis is a screening tool for clinical use. It should be used in conjunction with clinical interview and professional judgment."
    )

    html_content += f"""
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>This report is automatically generated by PsyNLP System | © 2025 Mental Health Analysis Tool</p>
            <p style="margin-top: 5px; font-size: 12px;">
                ⚠️ {disclaimer}
            </p>
        </div>
    </div>
</body>
</html>
"""

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ HTML report generated: {output_path}")
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
        print(f"✅ PDF report generated: {output_path}")

        return output_path
    except ImportError:
        print("❌ Unable to generate PDF: please install weasyprint")
        print("   Installation: pip install weasyprint")
        print("   Or use browser to print HTML to PDF")
        return None


def open_in_browser(html_path: str):
    """Open HTML report in default browser"""
    import webbrowser
    import os

    abs_path = os.path.abspath(html_path)
    webbrowser.open(f"file://{abs_path}")
    print(f"🌐 Report opened in browser")


# Example usage
if __name__ == "__main__":
    from modules.predict_score import calculate_article_severity

    # Analyze article
    result = calculate_article_severity("article.md")

    # Generate user report
    print("\n=== Generating User Report ===")
    html_user = generate_html_report(result, "report_user.html", mode="user")

    # Generate professional report
    print("\n=== Generating Professional Report ===")
    html_prof = generate_html_report(
        result, "report_professional.html", mode="professional"
    )

    # Open user report
    open_in_browser(html_user)
