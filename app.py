import streamlit as st
from modules import file_reader, visualizer
import os
from datetime import datetime

from modules import predict_score


st.set_page_config(
    page_title="PsyNLP - Mental Health Text Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #667eea;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


st.markdown('<h1 class="main-header">🧠 PsyNLP</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Mental Health Text Analysis Tool - Powered by NLP & LLM</p>', unsafe_allow_html=True)

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'report_path' not in st.session_state:
    st.session_state.report_path = None
if 'html_path' not in st.session_state:
    st.session_state.html_path = None
if 'pdf_available' not in st.session_state:
    st.session_state.pdf_available = False
if 'pdf_error' not in st.session_state:
    st.session_state.pdf_error = None


with st.sidebar:
    st.header("ℹ️ About PsyNLP")
    st.markdown("""
    **PsyNLP** is a local mental health text analysis tool that combines:
    - 🤖 BERT emotion detection
    - 🧠 DeBERTa mental health classification
    - 📊 LLM-based severity scoring
    - 📈 Comprehensive HTML reports
    
    **Privacy First**: All analysis happens locally on your device.
    """)
    st.divider()
    st.header("📋 Supported Formats")
    st.markdown("""
    - 📄 `.txt` - Plain text
    - 📝 `.md` - Markdown
    - 📃 `.docx` - Word documents
    """)
    st.divider()


st.header("1. Select User Mode")


user_mode = st.radio(
    "Choose your mode:",
    ("General User", "Professionals/Researchers"),
    index=0,
    help="**General User**: Get personalized recommendations and wellness tips.\n\n**Professionals/Researchers**: Access clinical insights, DSM-5 considerations, and risk assessments.",
    horizontal=True
)


mode_key = "user" if user_mode == "General User" else "professional"


if user_mode == "General User":
    st.info("👤 **General User Mode**: You'll receive supportive recommendations, coping strategies, and wellness guidance.")
else:
    st.info("🏥 **Professional Mode**: You'll receive clinical assessments, DSM-5 considerations, treatment recommendations, and risk evaluations.")

st.divider()


st.header("2. Upload Your File")


uploaded_file = st.file_uploader(
    "Choose a file to analyze",
    type=["md", "docx", "txt"],
    help="Upload your text file (.txt), markdown file (.md), or Word document (.docx)"
)

if uploaded_file:
    try:
        content = file_reader.read_file(uploaded_file)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 File Name", uploaded_file.name)
        with col2:
            st.metric("📊 File Size", f"{len(content)} chars")
        with col3:
            st.metric("📝 Word Count", f"{len(content.split())} words")
        st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
        st.divider()
        st.header("3. Run Analysis")
        if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing text... This may take a few minutes depending on file length."):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text("📝 Processing text and splitting into sentences...")
                    progress_bar.progress(20)
                    status_text.text("🤖 Running BERT emotion and mental health classification...")
                    progress_bar.progress(40)
                    status_text.text("🧠 Running LLM severity analysis on each sentence...")
                    progress_bar.progress(60)
                    result = predict_score.calculate_text_severity(content)
                    st.session_state.analysis_result = result
                    progress_bar.progress(80)
                    status_text.text("📊 Generating PDF report...")
                    report_dir = "reports"
                    os.makedirs(report_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    pdf_filename = f"report_{mode_key}_{timestamp}.pdf"
                    pdf_path = os.path.join(report_dir, pdf_filename)
                    html_filename = f"report_{mode_key}_{timestamp}.html"
                    html_path = os.path.join(report_dir, html_filename)
                    try:
                        visualizer.generate_pdf_report(
                            result,
                            output_path=pdf_path,
                            language="en",
                            mode=mode_key
                        )
                        st.session_state.report_path = pdf_path
                        st.session_state.html_path = html_path
                        st.session_state.pdf_available = True
                    except Exception as pdf_error:
                        status_text.text("⚠️ PDF generation failed, creating HTML report...")
                        visualizer.generate_html_report(
                            result,
                            output_path=html_path,
                            language="en",
                            mode=mode_key
                        )
                        st.session_state.report_path = html_path
                        st.session_state.html_path = html_path
                        st.session_state.pdf_available = False
                        st.session_state.pdf_error = str(pdf_error)
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    st.session_state.analysis_complete = True
                    import time
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    st.success("🎉 Analysis completed successfully!")
                    st.balloons()
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.exception(e)
                    st.session_state.analysis_complete = False

        if st.session_state.analysis_complete and st.session_state.analysis_result:
            st.divider()
            st.header("4. Analysis Results")
            result = st.session_state.analysis_result
            stats = result['severity_stats']
            nlp_analysis = result['nlp_analysis']
            overall_severity = stats['overall_severity_score']
            if overall_severity >= 7:
                st.markdown(f"""
                <div class="danger-box">
                    <h3>⚠️ HIGH RISK DETECTED</h3>
                    <p>Overall Severity Score: <strong>{overall_severity}/10</strong></p>
                    <p>This text shows significant mental health concerns. Professional help is strongly recommended.</p>
                </div>
                """, unsafe_allow_html=True)
            elif overall_severity >= 4:
                st.markdown(f"""
                <div class="warning-box">
                    <h3>⚠️ MODERATE RISK</h3>
                    <p>Overall Severity Score: <strong>{overall_severity}/10</strong></p>
                    <p>This text shows some mental health concerns. Consider seeking support.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-box">
                    <h3>✅ LOW RISK</h3>
                    <p>Overall Severity Score: <strong>{overall_severity}/10</strong></p>
                    <p>This text shows minimal mental health concerns.</p>
                </div>
                """, unsafe_allow_html=True)

            st.subheader("📊 Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Overall Severity",
                    f"{stats['overall_severity_score']}/10",
                    help="Comprehensive severity score"
                )
            with col2:
                st.metric(
                    "Maximum Risk",
                    f"{stats['max_severity']}/10",
                    help="Highest severity found in any sentence"
                )
            with col3:
                st.metric(
                    "Average Severity",
                    f"{stats['avg_severity']}/10",
                    help="Average severity across all sentences"
                )
            with col4:
                st.metric(
                    "High-Risk Sentences",
                    stats['high_risk_sentences'],
                    help="Number of sentences with severity ≥ 6"
                )

            st.subheader("🤖 NLP Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Emotion Detection</h4>
                    <p><strong>Label:</strong> {nlp_analysis['emotion_pred']}</p>
                    <p><strong>Confidence:</strong> {nlp_analysis['emotion_score']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Mental Health Classification</h4>
                    <p><strong>Label:</strong> {nlp_analysis['psy_pred']}</p>
                    <p><strong>Confidence:</strong> {nlp_analysis['psy_score']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)

            st.divider()
            st.subheader("📥 Download Full Report")
            if st.session_state.report_path and os.path.exists(st.session_state.report_path):
                if st.session_state.pdf_available:
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        with open(st.session_state.report_path, 'rb') as pdf_file:
                            pdf_bytes = pdf_file.read()
                        st.download_button(
                            label="📑 Download PDF Report",
                            data=pdf_bytes,
                            file_name=os.path.basename(st.session_state.report_path),
                            mime="application/pdf",
                            type="primary",
                            use_container_width=True
                        )
                    with col2:
                        if st.session_state.html_path and os.path.exists(st.session_state.html_path):
                            with open(st.session_state.html_path, 'r', encoding='utf-8') as html_file:
                                html_content = html_file.read()
                            st.download_button(
                                label="📄 Download HTML (Alternative)",
                                data=html_content,
                                file_name=os.path.basename(st.session_state.html_path),
                                mime="text/html",
                                use_container_width=True
                            )
                    with col3:
                        if st.button("🔄 Reset", use_container_width=True):
                            st.session_state.analysis_complete = False
                            st.session_state.analysis_result = None
                            st.session_state.report_path = None
                            st.session_state.html_path = None
                            st.session_state.pdf_available = False
                            st.rerun()
                    st.info("💡 **Tip**: PDF report is ready for download and printing. HTML version also available for web viewing.")
                else:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        with open(st.session_state.report_path, 'r', encoding='utf-8') as html_file:
                            html_content = html_file.read()
                        st.download_button(
                            label="📄 Download HTML Report",
                            data=html_content,
                            file_name=os.path.basename(st.session_state.report_path),
                            mime="text/html",
                            type="primary",
                            use_container_width=True
                        )
                    with col2:
                        if st.button("🔄 Reset", use_container_width=True):
                            st.session_state.analysis_complete = False
                            st.session_state.analysis_result = None
                            st.session_state.report_path = None
                            st.session_state.html_path = None
                            st.session_state.pdf_available = False
                            st.rerun()
                    st.warning("⚠️ PDF generation not available. HTML report provided instead.")
                    if st.session_state.pdf_error:
                        with st.expander("🔧 How to enable PDF generation"):
                            st.code(st.session_state.pdf_error)
                            st.markdown("""
                            **Install WeasyPrint for PDF support:**
                            
                            macOS:
                            ```bash
                            brew install cairo pango gdk-pixbuf libffi
                            pip install weasyprint
                            ```
                            
                            Ubuntu/Debian:
                            ```bash
                            sudo apt-get install libcairo2 libpango-1.0-0
                            pip install weasyprint
                            ```
                            """)
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        st.exception(e)
else:
    st.info("👆 Please upload a file to begin analysis.")

st.divider()
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.85rem; margin-top: 2rem;'>
    <p>🧠 <strong>PsyNLP</strong> - Mental Health Text Analysis Tool</p>
    <p>For educational and research purposes only | Not a substitute for professional care</p>
    <p><strong>Crisis Resources:</strong> 988 Suicide & Crisis Lifeline | Text 'HELLO' to 741741</p>
</div>
""", unsafe_allow_html=True)


