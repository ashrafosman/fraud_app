"""
Fraud Detection Agent - Streamlit App
PRODUCTION VERSION - For Databricks Apps deployment

Pattern based on databricks-ai-ticket-vectorsearch project
"""

import streamlit as st
import os
from databricks.sdk import WorkspaceClient

# Page configuration
st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark tech theme matching the design
st.markdown("""
<style>
    /* Dark tech theme colors */
    :root {
        --bg-dark: #0A1929;
        --bg-card: #132F4C;
        --cyan-primary: #00D9FF;
        --cyan-secondary: #00CCA3;
        --accent-yellow: #FFD93D;
        --text-primary: #E7EBF0;
        --text-secondary: #B2BAC2;
        --border-color: #1E4976;
    }
    
    /* Main app background */
    .stApp {
        background-color: #0A1929;
        color: #E7EBF0;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar styling - dark theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e2238 0%, #0A1929 100%);
        border-right: 1px solid #1E4976;
    }
    
    [data-testid="stSidebar"] * {
        color: #E7EBF0 !important;
    }
    
    [data-testid="stSidebar"] h1 {
        color: #00D9FF !important;
    }
    
    /* Dark cards */
    .tech-card {
        background: linear-gradient(135deg, #132F4C 0%, #0e2238 100%);
        border: 1px solid #1E4976;
        border-radius: 15px;
        padding: 25px;
        color: #E7EBF0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .tech-card:hover {
        border-color: #00D9FF;
        box-shadow: 0 8px 35px rgba(0,217,255,0.3);
        transform: translateY(-3px);
    }
    
    /* Cyan gradient text */
    .gradient-text {
        background: linear-gradient(90deg, #00D9FF 0%, #00CCA3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem;
        margin-bottom: 10px;
        text-shadow: 0 0 30px rgba(0,217,255,0.3);
    }
    
    /* Metric cards with dark theme */
    [data-testid="stMetric"] {
        background: #132F4C;
        border: 1px solid #1E4976;
        border-radius: 12px;
        padding: 15px;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #00D9FF !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #B2BAC2 !important;
    }
    
    /* Buttons - cyan theme */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 12px 30px;
        border: 2px solid #00D9FF;
        background: linear-gradient(135deg, #00D9FF 0%, #00CCA3 100%);
        color: #0A1929;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,217,255,0.3);
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 25px rgba(0,217,255,0.5);
        border-color: #00CCA3;
    }
    
    /* Text inputs and text areas */
    .stTextInput > div > div, .stTextArea > div > div {
        background: #132F4C;
        border: 1px solid #1E4976;
        color: #E7EBF0;
        border-radius: 10px;
    }
    
    .stTextInput > div > div:focus-within, .stTextArea > div > div:focus-within {
        border-color: #00D9FF;
        box-shadow: 0 0 10px rgba(0,217,255,0.3);
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: #132F4C;
        border: 1px solid #1E4976;
        color: #E7EBF0;
        border-radius: 10px;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: #132F4C;
        border: 1px solid #1E4976;
        border-radius: 10px;
        color: #E7EBF0;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #00D9FF;
    }
    
    /* Code blocks */
    .stCode {
        background: #0e2238 !important;
        border: 1px solid #1E4976 !important;
    }
    
    /* Success/Warning/Error boxes */
    .stSuccess, .stWarning, .stError, .stInfo {
        background: #132F4C;
        border-left: 4px solid #00D9FF;
        color: #E7EBF0;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #00D9FF !important;
    }
    
    /* Regular text */
    p, li, span {
        color: #E7EBF0 !important;
    }
    
    /* Dividers */
    hr {
        border-color: #1E4976;
    }
    
    /* Icon containers */
    .icon-container {
        font-size: 3rem;
        margin-bottom: 15px;
        filter: drop-shadow(0 0 10px rgba(0,217,255,0.3));
    }
    
    /* Status badge */
    .status-online {
        color: #00CCA3;
        font-weight: 700;
    }
    
    .status-error {
        color: #FF6B6B;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Configuration (read from environment variables)
CATALOG = os.getenv("CATALOG_NAME", "fraud_detection_dev")
SCHEMA = os.getenv("SCHEMA_NAME", "claims_analysis")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "148ccb90800933a1")
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")

# Initialize Databricks client (uses Databricks Apps authentication)
@st.cache_resource
def get_workspace_client():
    """Initialize Databricks WorkspaceClient (automatically authenticated in Databricks Apps)"""
    try:
        return WorkspaceClient()
    except Exception as e:
        st.error(f"Failed to initialize Databricks client: {e}")
        return None

w = get_workspace_client()

# Sidebar
st.sidebar.markdown("""
<div style='text-align: center; margin-bottom: 30px; padding: 20px 0;'>
    <h1 style='color: #00D9FF; font-size: 1.5rem; margin: 0;'>🛡️ Fraud Detection</h1>
    <p style='color: #B2BAC2; font-size: 0.85rem; margin: 5px 0 0 0;'>AI-POWERED CLAIMS ANALYSIS</p>
</div>
""", unsafe_allow_html=True)

# Connection status
status_color = "#00CCA3" if w else "#FF6B6B"
status_icon = "🟢" if w else "❌"
status_text = "Connected" if w else "Not connected to Databricks"

st.sidebar.markdown(f"""
<div style='background: #132F4C; border: 1px solid #1E4976; padding: 12px; border-radius: 10px; margin-bottom: 20px; text-align: center;'>
    <p style='margin: 0; color: {status_color}; font-weight: 700; font-size: 0.9rem;'>{status_icon} {status_text}</p>
</div>
""", unsafe_allow_html=True)

# Environment info
st.sidebar.markdown(f"""
<div style='background: rgba(0,217,255,0.05); border: 1px solid #1E4976; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
    <p style='margin: 0; font-size: 0.75rem; color: #B2BAC2; text-transform: uppercase; letter-spacing: 1px;'>🌍 Environment</p>
    <p style='margin: 5px 0 0 0; font-size: 0.9rem; color: #E7EBF0;'><b>{ENVIRONMENT.upper()}</b></p>
    <hr style='margin: 10px 0; border-color: #1E4976;'>
    <p style='margin: 5px 0; font-size: 0.8rem; color: #B2BAC2;'>📊 Catalog: <span style='color: #00D9FF;'>{CATALOG}</span></p>
    <p style='margin: 5px 0; font-size: 0.8rem; color: #B2BAC2;'>📁 Schema: <span style='color: #00D9FF;'>{SCHEMA}</span></p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<hr style='border-color: #1E4976; margin: 20px 0;'>", unsafe_allow_html=True)

# Navigation menu
st.sidebar.markdown("""
<div style='margin-top: 20px;'>
    <p style='color: #00D9FF; font-weight: 700; margin-bottom: 15px; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px;'>📍 NAVIGATION</p>
    <div style='margin-left: 10px;'>
        <p style='margin: 12px 0; color: #E7EBF0; font-size: 0.95rem;'>🏠  Home - Dashboard & Overview</p>
        <p style='margin: 12px 0; color: #E7EBF0; font-size: 0.95rem;'>📊  Claim Analysis - AI Agent Analysis</p>
        <p style='margin: 12px 0; color: #E7EBF0; font-size: 0.95rem;'>⚡  Batch Processing - Bulk Claims</p>
        <p style='margin: 12px 0; color: #E7EBF0; font-size: 0.95rem;'>📈  Fraud Insights - Analytics & Trends</p>
        <p style='margin: 12px 0; color: #E7EBF0; font-size: 0.95rem;'>🔍  Case Search - Similar Cases</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Databricks header badge
st.markdown("""
<div style='text-align: center; margin: 20px 0 30px 0;'>
    <span style='background: linear-gradient(90deg, #00D9FF 0%, #00CCA3 100%); padding: 8px 20px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; color: #0A1929; letter-spacing: 2px;'>
        ◆ DATABRICKS ◆
    </span>
</div>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div style='text-align: center; padding: 20px 0 30px 0;'>
    <h1 class='gradient-text'>AI-Powered Fraud Detection System</h1>
    <p style='font-size: 1.3rem; color: #B2BAC2; margin-top: 10px;'>Intelligent Insurance Claims Analysis Platform</p>
    <div style='margin-top: 20px;'>
        <span style='color: #FFD93D; margin: 0 10px;'>⚡ LangGraph Agents</span>
        <span style='color: #B2BAC2;'>•</span>
        <span style='color: #00CCA3; margin: 0 10px;'>🎯 Unity Catalog</span>
        <span style='color: #B2BAC2;'>•</span>
        <span style='color: #00D9FF; margin: 0 10px;'>🔍 Vector Search</span>
        <span style='color: #B2BAC2;'>•</span>
        <span style='color: #B2BAC2; margin: 0 10px;'>💬 Genie AI</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Feature Cards Section
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class='tech-card' style='text-align: center;'>
        <div style='font-size: 3rem; margin-bottom: 15px;'>🧠</div>
        <h3 style='color: #FFD93D; margin: 10px 0; font-size: 1.1rem;'>LANGGRAPH AGENTS</h3>
        <p style='color: #B2BAC2; font-size: 0.85rem; margin: 0;'>ReAct pattern reasoning</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='tech-card' style='text-align: center;'>
        <div style='font-size: 3rem; margin-bottom: 15px;'>🎯</div>
        <h3 style='color: #00D9FF; margin: 10px 0; font-size: 1.1rem;'>UC AI FUNCTIONS</h3>
        <p style='color: #B2BAC2; font-size: 0.85rem; margin: 0;'>Classify, Extract, Explain</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='tech-card' style='text-align: center;'>
        <div style='font-size: 3rem; margin-bottom: 15px;'>🔍</div>
        <h3 style='color: #00CCA3; margin: 10px 0; font-size: 1.1rem;'>VECTOR SEARCH</h3>
        <p style='color: #B2BAC2; font-size: 0.85rem; margin: 0;'>Find similar cases</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class='tech-card' style='text-align: center;'>
        <div style='font-size: 3rem; margin-bottom: 15px;'>💬</div>
        <h3 style='color: #FFD93D; margin: 10px 0; font-size: 1.1rem;'>GENIE API</h3>
        <p style='color: #B2BAC2; font-size: 0.85rem; margin: 0;'>Natural language queries</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Quick Start Section
st.markdown("""
<div style='margin: 40px 0 20px 0;'>
    <h2 style='color: #00D9FF; font-size: 2rem; margin-bottom: 5px;'>🚀 Quick Start</h2>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div style='background: #00D9FF; padding: 20px; border-radius: 12px; text-align: center; cursor: pointer; transition: all 0.3s;'>
        <div style='color: #0A1929; font-weight: 700; font-size: 1rem;'>📊 ANALYZE</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background: #132F4C; border: 1px solid #1E4976; padding: 20px; border-radius: 12px; text-align: center; transition: all 0.3s;'>
        <div style='color: #00D9FF; font-weight: 700; font-size: 1rem;'>⚡ BATCH</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='background: #132F4C; border: 1px solid #1E4976; padding: 20px; border-radius: 12px; text-align: center; transition: all 0.3s;'>
        <div style='color: #00D9FF; font-weight: 700; font-size: 1rem;'>📈 INSIGHTS</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style='background: #132F4C; border: 1px solid #1E4976; padding: 20px; border-radius: 12px; text-align: center; transition: all 0.3s;'>
        <div style='color: #00D9FF; font-weight: 700; font-size: 1rem;'>🔍 SEARCH</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div style='background: #132F4C; border: 1px solid #1E4976; padding: 20px; border-radius: 12px; text-align: center; transition: all 0.3s;'>
        <div style='color: #FFD93D; font-weight: 700; font-size: 1rem;'>📁 DATA</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# System Status
st.markdown("""
<div style='margin: 40px 0 20px 0;'>
    <h2 style='color: #00D9FF; font-size: 2rem; margin-bottom: 20px;'>📊 System Status</h2>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    status_color = "#00CCA3" if w else "#FF6B6B"
    status_text = "Active" if w else "Error"
    status_detail = "✓ Latest" if w else "✗ Check Connection"
    st.markdown(f"""
    <div style='background: #132F4C; border: 1px solid #1E4976; padding: 20px; border-radius: 12px;'>
        <p style='color: #B2BAC2; font-size: 0.75rem; margin: 0; text-transform: uppercase; letter-spacing: 1px;'>🌍 ENVIRONMENT</p>
        <p style='color: {status_color}; font-size: 1.8rem; font-weight: 700; margin: 10px 0;'>{status_text}</p>
        <p style='color: {status_color}; font-size: 0.8rem; margin: 0;'>{status_detail}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background: #132F4C; border: 1px solid #1E4976; padding: 20px; border-radius: 12px;'>
        <p style='color: #B2BAC2; font-size: 0.75rem; margin: 0; text-transform: uppercase; letter-spacing: 1px;'>🤖 LLM MODEL</p>
        <p style='color: #00D9FF; font-size: 1.5rem; font-weight: 700; margin: 10px 0;'>Claude Sonnet 4.5</p>
        <p style='color: #00CCA3; font-size: 0.8rem; margin: 0;'>✓ Latest</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='background: #132F4C; border: 1px solid #1E4976; padding: 20px; border-radius: 12px;'>
        <p style='color: #B2BAC2; font-size: 0.75rem; margin: 0; text-transform: uppercase; letter-spacing: 1px;'>🔧 AI TOOLS</p>
        <p style='color: #00D9FF; font-size: 1.8rem; font-weight: 700; margin: 10px 0;'>4</p>
        <p style='color: #00CCA3; font-size: 0.8rem; margin: 0;'>✓ Classify, Extract, Search, Query</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    system_status_color = "#FF6B6B" if not w else "#00CCA3"
    system_status_text = "Error" if not w else "Online"
    system_status_detail = "✗ Check Connection" if not w else "✓ All Systems Operational"
    st.markdown(f"""
    <div style='background: #132F4C; border: 1px solid #1E4976; padding: 20px; border-radius: 12px;'>
        <p style='color: #B2BAC2; font-size: 0.75rem; margin: 0; text-transform: uppercase; letter-spacing: 1px;'>⚡ SYSTEM</p>
        <p style='color: {system_status_color}; font-size: 1.8rem; font-weight: 700; margin: 10px 0;'>{system_status_text}</p>
        <p style='color: {system_status_color}; font-size: 0.8rem; margin: 0;'>{system_status_detail}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; padding: 40px 20px; background: linear-gradient(135deg, #132F4C 0%, #0e2238 100%); border: 1px solid #1E4976; border-radius: 15px; margin-top: 60px;'>
    <h3 style='color: #00D9FF; margin: 0 0 15px 0; font-size: 1.8rem;'>Ready to Detect Fraud? 🚀</h3>
    <p style='margin: 0; font-size: 1.1rem; color: #B2BAC2;'>Select a page from the sidebar to begin analyzing claims with AI</p>
</div>
""", unsafe_allow_html=True)


