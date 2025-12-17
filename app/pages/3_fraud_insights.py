"""
Fraud Insights Dashboard
Displays analytics and trends from batch fraud analysis results
"""

import streamlit as st
import os
from databricks import sql
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# Page configuration
st.set_page_config(
    page_title="Fraud Insights | Fraud Detection",
    page_icon="📈",
    layout="wide"
)

# Dark theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0A1929;
        color: #E7EBF0;
    }
    
    /* Sidebar dark theme */
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
    
    h1, h2, h3, h4, h5, h6 {
        color: #00D9FF !important;
    }
    p, li, span, label {
        color: #E7EBF0 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00D9FF 0%, #00CCA3 100%);
        color: #0A1929;
        border: 2px solid #00D9FF;
        font-weight: 600;
        border-radius: 10px;
        padding: 12px 30px;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 25px rgba(0,217,255,0.5);
    }
    [data-testid="stMetric"] {
        background: #132F4C;
        border: 1px solid #1E4976;
        border-radius: 12px;
        padding: 15px;
    }
    [data-testid="stMetricValue"] {
        color: #00D9FF !important;
    }
    .stSelectbox > div > div {
        background: #132F4C;
        border: 1px solid #1E4976;
        color: #E7EBF0;
    }
    /* Plotly charts dark theme */
    .js-plotly-plot {
        background: #132F4C !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color: #00D9FF;'>📈 Fraud Insights</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #B2BAC2; font-size: 1.1rem;'>Analytics and trends from fraud detection analysis</p>", unsafe_allow_html=True)

# Configuration
CATALOG = os.getenv("CATALOG_NAME", "fraud_detection_dev")
SCHEMA = os.getenv("SCHEMA_NAME", "claims_analysis")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID")

# Initialize clients
@st.cache_resource
def get_workspace_client():
    """Initialize Databricks WorkspaceClient"""
    try:
        return WorkspaceClient()
    except Exception as e:
        st.error(f"Failed to initialize Databricks client: {e}")
        return None

# Initialize clients
@st.cache_resource
def get_workspace_client():
    """Initialize Databricks WorkspaceClient"""
    try:
        return WorkspaceClient()
    except Exception as e:
        st.error(f"Failed to initialize Databricks client: {e}")
        return None

def get_sql_connection():
    """Create Databricks SQL connection - called lazily when needed"""
    try:
        cfg = Config()
        return sql.connect(
            server_hostname=cfg.host,
            http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
            credentials_provider=lambda: cfg.authenticate,
        )
    except Exception as e:
        st.error(f"SQL Connection error: {e}")
        return None

w = get_workspace_client()

# Get Genie Space ID - Try multiple sources in order
def get_genie_space_id():
    """
    Get Genie Space ID from multiple sources (in priority order):
    1. Environment variable (if set in app.yaml)
    2. config_genie table (automatic lookup)
    
    This allows automatic discovery without manual config updates.
    """
    # Try environment variable first
    env_genie_id = os.getenv("GENIE_SPACE_ID")
    if env_genie_id:
        return env_genie_id
    
    # Fall back to querying config_genie table
    try:
        sql_conn = get_sql_connection()
        if sql_conn:
            with sql_conn.cursor() as cursor:
                cursor.execute(f"""
                    SELECT config_value 
                    FROM {CATALOG}.{SCHEMA}.config_genie 
                    WHERE config_key = 'genie_space_id'
                """)
                result = cursor.fetchone()
                if result and result[0]:
                    return result[0]
    except Exception as e:
        # Silently fail - will show warning in UI
        pass
    
    return None

GENIE_SPACE_ID = get_genie_space_id()

# SQL Query Functions
@st.cache_data(ttl=300)
def get_fraud_statistics():
    """Get overall fraud statistics"""
    sql_conn = get_sql_connection()
    if not sql_conn:
        return None
    
    try:
        with sql_conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_claims,
                    SUM(CASE WHEN is_fraudulent THEN 1 ELSE 0 END) as fraud_cases,
                    ROUND(AVG(CASE WHEN is_fraudulent THEN 1.0 ELSE 0.0 END) * 100, 2) as fraud_rate,
                    ROUND(AVG(risk_score), 2) as avg_risk_score
                FROM {CATALOG}.{SCHEMA}.fraud_analysis
            """)
            result = cursor.fetchone()
            if result:
                return {
                    "total_claims": result[0],
                    "fraud_cases": result[1],
                    "fraud_rate": result[2],
                    "avg_risk_score": result[3]
                }
    except Exception as e:
        st.error(f"Error fetching statistics: {e}")
    return None

@st.cache_data(ttl=300)
def get_fraud_by_type():
    """Get fraud breakdown by type"""
    sql_conn = get_sql_connection()
    if not sql_conn:
        return None
    
    try:
        with sql_conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT 
                    fraud_type,
                    COUNT(*) as count
                FROM {CATALOG}.{SCHEMA}.fraud_analysis
                WHERE is_fraudulent = TRUE
                GROUP BY fraud_type
                ORDER BY count DESC
            """)
            results = cursor.fetchall()
            if results:
                return pd.DataFrame(results, columns=["Fraud Type", "Count"])
    except Exception as e:
        st.error(f"Error fetching fraud types: {e}")
    return None

@st.cache_data(ttl=300)
def get_top_indicators():
    """Get top fraud indicators"""
    sql_conn = get_sql_connection()
    if not sql_conn:
        return None
    
    try:
        with sql_conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT 
                    explode(red_flags) as indicator
                FROM {CATALOG}.{SCHEMA}.fraud_analysis
                WHERE is_fraudulent = TRUE AND red_flags IS NOT NULL
            """)
            results = cursor.fetchall()
            if results:
                indicators = [r[0] for r in results]
                indicator_counts = pd.Series(indicators).value_counts().head(10)
                return pd.DataFrame({
                    "Indicator": indicator_counts.index,
                    "Count": indicator_counts.values
                })
    except Exception as e:
        st.error(f"Error fetching indicators: {e}")
    return None

@st.cache_data(ttl=300)
def get_fraud_trends():
    """Get fraud detection trends over time"""
    sql_conn = get_sql_connection()
    if not sql_conn:
        return None
    
    try:
        with sql_conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT 
                    DATE(analysis_timestamp) as date,
                    COUNT(*) as total_claims,
                    SUM(CASE WHEN is_fraudulent THEN 1 ELSE 0 END) as fraud_cases
                FROM {CATALOG}.{SCHEMA}.fraud_analysis
                GROUP BY DATE(analysis_timestamp)
                ORDER BY date
            """)
            results = cursor.fetchall()
            if results:
                return pd.DataFrame(results, columns=["Date", "Total Claims", "Fraud Cases"])
    except Exception as e:
        st.error(f"Error fetching trends: {e}")
    return None

# Main Dashboard
st.markdown("---")

# Key Metrics
stats = get_fraud_statistics()
if stats:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Claims Analyzed",
            f"{stats['total_claims']:,}",
            help="Total number of claims processed"
        )
    
    with col2:
        st.metric(
            "Fraud Cases Detected",
            f"{stats['fraud_cases']:,}",
            help="Number of fraudulent claims identified"
        )
    
    with col3:
        st.metric(
            "Fraud Detection Rate",
            f"{stats['fraud_rate']}%",
            help="Percentage of claims flagged as fraudulent"
        )
    
    with col4:
        st.metric(
            "Average Risk Score",
            f"{stats['avg_risk_score']}/10",
            help="Mean risk score across all claims"
        )

st.markdown("---")

# Charts Section
st.subheader("📊 Fraud Analytics")

col1, col2 = st.columns(2)

with col1:
    # Fraud Type Breakdown
    st.markdown("### Fraud Type Breakdown")
    fraud_types = get_fraud_by_type()
    if fraud_types is not None and not fraud_types.empty:
        fig = px.pie(
            fraud_types,
            values="Count",
            names="Fraud Type",
            title="Distribution of Fraud Types",
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No fraud type data available yet.")

with col2:
    # Top Fraud Indicators
    st.markdown("### Top Fraud Red Flags")
    indicators = get_top_indicators()
    if indicators is not None and not indicators.empty:
        fig = px.bar(
            indicators,
            x="Count",
            y="Indicator",
            orientation='h',
            title="Most Common Fraud Indicators",
            labels={"Count": "Occurrences", "Indicator": "Red Flag"}
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No fraud indicator data available yet.")

# Fraud Trends Over Time
st.markdown("### 📈 Fraud Detection Trends Over Time")
trends = get_fraud_trends()
if trends is not None and not trends.empty:
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=trends["Date"],
        y=trends["Total Claims"],
        name="Total Claims",
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=trends["Date"],
        y=trends["Fraud Cases"],
        name="Fraud Cases",
        line=dict(color='red', width=2),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title="Claims Analysis Over Time",
        xaxis_title="Date",
        yaxis_title="Number of Claims",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No trend data available yet. Process more claims to see trends.")

st.markdown("---")

# Genie Natural Language Interface
st.subheader("💬 Ask Genie - Natural Language Queries")

if GENIE_SPACE_ID:
    st.markdown("""
    Ask questions about your fraud data in plain English. Genie will automatically generate SQL and return results.
    """)
    
    # Expanded example questions
    example_questions = [
        "Show me all fraudulent claims",
        "What are the top 10 highest risk claims?",
        "Show claims with amount greater than $40,000",
        "Which providers have the most fraud cases?",
        "Show fraud trends by month",
        "What is the average claim amount for fraudulent vs legitimate claims?",
        "Which claim types have the highest fraud rate?",
        "Show me claims with risk score above 8",
        "What are the most common fraud red flags?",
        "Compare fraud rates across different claim types"
    ]
    
    # User input
    user_question = st.text_input(
        "Your question:",
        placeholder="e.g., Show me all high-risk fraudulent claims",
        help="Ask any question about your fraud detection data"
    )
    
    # Quick question buttons in a grid
    st.markdown("**Quick Questions:**")
    col1, col2, col3 = st.columns(3)
    
    for i, question in enumerate(example_questions):
        col_idx = i % 3
        with [col1, col2, col3][col_idx]:
            if st.button(question, key=f"q_{i}", use_container_width=True):
                user_question = question
    
    if user_question:
        with st.spinner("🤔 Genie is thinking..."):
            try:
                # Start conversation using official Genie API pattern
                start_response = w.api_client.do(
                    'POST',
                    f'/api/2.0/genie/spaces/{GENIE_SPACE_ID}/start-conversation',
                    body={'content': user_question}
                )
                
                conversation_id = start_response.get('conversation_id')
                message_id = start_response.get('message_id')
                
                if not conversation_id or not message_id:
                    st.error("Failed to start Genie conversation")
                else:
                    # Poll for result
                    max_attempts = 30  # 30 * 2 = 60 seconds max
                    attempt = 0
                    
                    while attempt < max_attempts:
                        time.sleep(2)  # Wait 2 seconds between polls
                        attempt += 1
                        
                        # Get message status
                        message_response = w.api_client.do(
                            'GET',
                            f'/api/2.0/genie/spaces/{GENIE_SPACE_ID}/conversations/{conversation_id}/messages/{message_id}'
                        )
                        
                        status = message_response.get('status')
                        
                        if status == 'COMPLETED':
                            # Extract results
                            attachments = message_response.get('attachments', [])
                            
                            if attachments:
                                attachment = attachments[0]
                                text_response = attachment.get('text', {}).get('content', '')
                                query = attachment.get('query', {}).get('query', '')
                                
                                # Display text response
                                if text_response:
                                    st.success("**Genie's Response:**")
                                    st.markdown(text_response)
                                
                                # Display generated SQL
                                if query:
                                    with st.expander("🔍 View Generated SQL"):
                                        st.code(query, language="sql")
                                    
                                    # Get query results
                                    try:
                                        attachment_id = attachment.get('query', {}).get('attachment_id') or attachment.get('attachment_id')
                                        if attachment_id:
                                            result_response = w.api_client.do(
                                                'GET',
                                                f'/api/2.0/genie/spaces/{GENIE_SPACE_ID}/conversations/{conversation_id}/messages/{message_id}/query-result/{attachment_id}'
                                            )
                                            
                                            # Extract data from statement_response (correct path from notebook)
                                            stmt_response = result_response.get('statement_response', {})
                                            
                                            if stmt_response:
                                                # Get schema from manifest
                                                manifest = stmt_response.get('manifest', {})
                                                schema = manifest.get('schema', {})
                                                columns = schema.get('columns', [])
                                                column_names = [col.get('name') for col in columns]
                                                
                                                # Get data rows from result
                                                result_obj = stmt_response.get('result', {})
                                                data_array = result_obj.get('data_array', [])
                                                
                                                if data_array and column_names:
                                                    # Create DataFrame
                                                    df = pd.DataFrame(data_array, columns=column_names)
                                                    
                                                    st.success(f"✅ Found {len(df)} results")
                                                    st.dataframe(df, use_container_width=True)
                                                    
                                                    # Auto-generate chart if applicable
                                                    if len(df.columns) == 2 and len(df) > 1 and len(df) < 50:
                                                        st.markdown("**📊 Visualization:**")
                                                        fig = px.bar(df, x=df.columns[0], y=df.columns[1])
                                                        st.plotly_chart(fig, use_container_width=True)
                                                else:
                                                    st.info("Query executed successfully but returned no results.")
                                            else:
                                                st.warning("No statement_response in query result")
                                    except Exception as e:
                                        st.warning(f"Could not fetch query results: {e}")
                            else:
                                st.info("Query completed but no results available.")
                            break
                            
                        elif status == 'FAILED':
                            error = message_response.get('error', {})
                            st.error(f"Query failed: {error}")
                            break
                        elif status == 'CANCELLED':
                            st.warning("Query was cancelled")
                            break
                    
                    if attempt >= max_attempts:
                        st.warning("Query timed out. Please try a simpler question.")
                    
            except Exception as e:
                st.error(f"Error executing Genie query: {e}")
                st.info("💡 Make sure you've granted **Can Run** permissions to the app's service principal on the Genie Space.")
else:
    st.warning("⚠️ Genie Space not configured. The Genie natural language interface is currently unavailable.")
    st.markdown("""
    **To enable Genie:**
    1. Genie Space is automatically created during setup
    2. Grant **Can Run** permission to your app's service principal on the Genie Space
    3. See README for detailed instructions
    """)

st.markdown("---")
st.caption("💡 **Tip:** Process more claims in Batch Processing to see richer insights and trends!")
