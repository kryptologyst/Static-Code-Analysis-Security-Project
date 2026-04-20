"""
Streamlit demo application for Static Code Analysis.

This module provides an interactive web interface for uploading code,
analyzing vulnerabilities, and viewing results with explanations.
"""

import logging
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Dict, List, Optional
import tempfile
import os
from pathlib import Path

# Import our modules
from src.analysis.scanner import CodeScanner, Vulnerability
from src.models.vulnerability_detector import VulnerabilityDetector
from src.data.features import CodeFeatureExtractor
from src.utils.explainability import ExplainabilityManager
from src.evaluation.metrics import VulnerabilityEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Static Code Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .vulnerability-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .critical { border-left: 5px solid #d62728; }
    .high { border-left: 5px solid #ff7f0e; }
    .medium { border-left: 5px solid #ffbb78; }
    .low { border-left: 5px solid #2ca02c; }
    .info { border-left: 5px solid #17a2b8; }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vulnerabilities' not in st.session_state:
    st.session_state.vulnerabilities = []
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'model' not in st.session_state:
    st.session_state.model = None


def load_config() -> Dict[str, Any]:
    """Load configuration."""
    return {
        'random_seed': 42,
        'max_file_size': 10 * 1024 * 1024,  # 10MB
        'supported_extensions': ['.py', '.txt'],
        'model_type': 'gradient_boosting'
    }


def initialize_components() -> tuple:
    """Initialize analysis components."""
    config = load_config()
    
    scanner = CodeScanner(config)
    detector = VulnerabilityDetector(config.get('model_type', 'gradient_boosting'))
    explainer = ExplainabilityManager(config)
    
    return scanner, detector, explainer


def display_header():
    """Display the main header."""
    st.markdown('<h1 class="main-header">🔍 Static Code Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Analyze Python code for security vulnerabilities and quality issues
        </p>
        <p style="font-size: 0.9rem; color: #999;">
            <strong>DISCLAIMER:</strong> This is a defensive research and educational tool. 
            Not intended for production security operations.
        </p>
    </div>
    """, unsafe_allow_html=True)


def display_sidebar():
    """Display the sidebar with options."""
    st.sidebar.title("Analysis Options")
    
    # Analysis type
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Rule-based", "ML-enhanced", "Both"],
        help="Choose the type of analysis to perform"
    )
    
    # Severity filter
    severity_filter = st.sidebar.multiselect(
        "Severity Filter",
        ["critical", "high", "medium", "low", "info"],
        default=["critical", "high", "medium"],
        help="Filter vulnerabilities by severity level"
    )
    
    # Show explanations
    show_explanations = st.sidebar.checkbox(
        "Show Explanations",
        value=True,
        help="Display detailed explanations for detected vulnerabilities"
    )
    
    # Show feature importance
    show_features = st.sidebar.checkbox(
        "Show Feature Importance",
        value=False,
        help="Display feature importance for ML predictions"
    )
    
    return {
        'analysis_type': analysis_type,
        'severity_filter': severity_filter,
        'show_explanations': show_explanations,
        'show_features': show_features
    }


def upload_code_section():
    """Handle code upload and input."""
    st.header("📁 Code Input")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Python file",
        type=['py', 'txt'],
        help="Upload a Python file to analyze"
    )
    
    # Text input
    st.subheader("Or paste code directly:")
    code_input = st.text_area(
        "Code to analyze",
        height=300,
        placeholder="Paste your Python code here...",
        help="Enter Python code directly for analysis"
    )
    
    # Sample code option
    if st.checkbox("Use sample vulnerable code"):
        code_input = """
def login(username, password):
    if password == 'admin123':  # Hardcoded password
        return True
    return False

def process_user_input(user_input):
    result = eval(user_input)  # Dangerous eval usage
    return result

def query_database(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"  # SQL injection
    return execute_query(query)

def delete_file(filename):
    os.system(f'rm {filename}')  # Shell injection
"""
        st.code(code_input, language='python')
    
    # Get the code to analyze
    code_to_analyze = None
    file_name = None
    
    if uploaded_file is not None:
        try:
            code_to_analyze = uploaded_file.read().decode('utf-8')
            file_name = uploaded_file.name
        except Exception as e:
            st.error(f"Error reading file: {e}")
    elif code_input.strip():
        code_to_analyze = code_input
        file_name = "input_code.py"
    
    return code_to_analyze, file_name


def analyze_code(code: str, file_name: str, options: Dict[str, Any]) -> List[Vulnerability]:
    """Analyze code for vulnerabilities."""
    scanner, detector, explainer = initialize_components()
    
    # Perform analysis
    with st.spinner("Analyzing code..."):
        vulnerabilities = scanner.scan_code(code, file_name)
    
    # Filter by severity
    if options['severity_filter']:
        vulnerabilities = [
            v for v in vulnerabilities 
            if v.severity in options['severity_filter']
        ]
    
    return vulnerabilities


def display_vulnerability_card(vuln: Vulnerability, show_explanations: bool, explainer: ExplainabilityManager):
    """Display a single vulnerability card."""
    severity_colors = {
        'critical': '#d62728',
        'high': '#ff7f0e', 
        'medium': '#ffbb78',
        'low': '#2ca02c',
        'info': '#17a2b8'
    }
    
    color = severity_colors.get(vuln.severity, '#666')
    
    with st.container():
        st.markdown(f"""
        <div class="vulnerability-card {vuln.severity}">
            <h4 style="color: {color}; margin-top: 0;">
                {vuln.severity.upper()}: {vuln.vulnerability_type.replace('_', ' ').title()}
            </h4>
            <p><strong>Line {vuln.line_number}:</strong> {vuln.description}</p>
            <code>{vuln.code_snippet}</code>
            <p><strong>Confidence:</strong> {vuln.confidence:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if show_explanations:
            with st.expander("📖 Explanation", expanded=False):
                explanation = explainer.explain_vulnerability(vuln.to_dict(), vuln.code_snippet)
                
                st.write(f"**Rule Evidence:**")
                for evidence in explanation.rule_evidence or []:
                    st.write(f"- {evidence}")
                
                st.write(f"**Explanation:** {explanation.explanation_text}")
                
                if explanation.feature_importance:
                    st.write("**Feature Importance:**")
                    for feature, importance in list(explanation.feature_importance.items())[:5]:
                        st.write(f"- {feature}: {importance:.3f}")


def display_analysis_results(vulnerabilities: List[Vulnerability], options: Dict[str, Any]):
    """Display analysis results."""
    if not vulnerabilities:
        st.success("✅ No vulnerabilities detected!")
        return
    
    st.header("🚨 Analysis Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Vulnerabilities", len(vulnerabilities))
    
    with col2:
        critical_count = sum(1 for v in vulnerabilities if v.severity == 'critical')
        st.metric("Critical", critical_count)
    
    with col3:
        high_count = sum(1 for v in vulnerabilities if v.severity == 'high')
        st.metric("High", high_count)
    
    with col4:
        avg_confidence = sum(v.confidence for v in vulnerabilities) / len(vulnerabilities)
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    # Severity distribution chart
    st.subheader("📊 Severity Distribution")
    severity_counts = {}
    for vuln in vulnerabilities:
        severity_counts[vuln.severity] = severity_counts.get(vuln.severity, 0) + 1
    
    if severity_counts:
        fig = px.pie(
            values=list(severity_counts.values()),
            names=list(severity_counts.keys()),
            title="Vulnerability Severity Distribution",
            color_discrete_map={
                'critical': '#d62728',
                'high': '#ff7f0e',
                'medium': '#ffbb78', 
                'low': '#2ca02c',
                'info': '#17a2b8'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Vulnerability types chart
    st.subheader("🔍 Vulnerability Types")
    type_counts = {}
    for vuln in vulnerabilities:
        type_counts[vuln.vulnerability_type] = type_counts.get(vuln.vulnerability_type, 0) + 1
    
    if type_counts:
        fig = px.bar(
            x=list(type_counts.keys()),
            y=list(type_counts.values()),
            title="Vulnerability Types",
            labels={'x': 'Vulnerability Type', 'y': 'Count'}
        )
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Individual vulnerabilities
    st.subheader("📋 Detailed Results")
    
    explainer = ExplainabilityManager()
    
    for vuln in vulnerabilities:
        display_vulnerability_card(vuln, options['show_explanations'], explainer)


def display_data_table(vulnerabilities: List[Vulnerability]):
    """Display vulnerabilities in a data table."""
    if not vulnerabilities:
        return
    
    st.subheader("📊 Data Table")
    
    # Convert to DataFrame
    data = []
    for vuln in vulnerabilities:
        data.append({
            'Line': vuln.line_number,
            'Severity': vuln.severity,
            'Type': vuln.vulnerability_type,
            'Description': vuln.description,
            'Confidence': vuln.confidence,
            'Code': vuln.code_snippet
        })
    
    df = pd.DataFrame(data)
    
    # Display with filtering
    st.dataframe(
        df,
        use_container_width=True,
        height=400
    )
    
    # Download option
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="vulnerability_analysis.csv",
        mime="text/csv"
    )


def display_model_info():
    """Display information about the ML model."""
    st.sidebar.header("🤖 Model Information")
    
    st.sidebar.info("""
    **Model Type:** Gradient Boosting Classifier
    
    **Features Used:**
    - AST metrics (nodes, depth, complexity)
    - Text features (line count, comments)
    - Security patterns (dangerous functions)
    - Code complexity metrics
    
    **Training Data:** Synthetic vulnerability dataset
    """)
    
    if st.sidebar.button("🔄 Retrain Model"):
        st.sidebar.info("Model retraining not implemented in demo")


def main():
    """Main application function."""
    display_header()
    
    # Initialize components
    scanner, detector, explainer = initialize_components()
    
    # Display sidebar
    options = display_sidebar()
    
    # Display model info
    display_model_info()
    
    # Code input section
    code_to_analyze, file_name = upload_code_section()
    
    # Analysis button
    if st.button("🔍 Analyze Code", type="primary"):
        if code_to_analyze:
            # Analyze code
            vulnerabilities = analyze_code(code_to_analyze, file_name, options)
            
            # Store in session state
            st.session_state.vulnerabilities = vulnerabilities
            
            # Display results
            display_analysis_results(vulnerabilities, options)
            display_data_table(vulnerabilities)
        else:
            st.warning("Please upload a file or enter code to analyze.")
    
    # Display previous results if available
    if st.session_state.vulnerabilities:
        st.header("📋 Previous Analysis Results")
        
        if st.button("🔄 Re-analyze"):
            if code_to_analyze:
                vulnerabilities = analyze_code(code_to_analyze, file_name, options)
                st.session_state.vulnerabilities = vulnerabilities
                st.rerun()
        
        display_analysis_results(st.session_state.vulnerabilities, options)
        display_data_table(st.session_state.vulnerabilities)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>Static Code Analysis Demo | Educational Research Tool</p>
        <p>⚠️ This tool is for educational purposes only. Always validate security findings manually.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
