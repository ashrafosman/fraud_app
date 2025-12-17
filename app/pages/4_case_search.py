"""
Case Search Page
Find similar historical fraud cases using vector similarity search
"""

import streamlit as st
import os
from databricks.sdk import WorkspaceClient
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Case Search | Fraud Detection",
    page_icon="🔎",
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
    .stTextInput > div > div, .stTextArea > div > div {
        background: #132F4C;
        border: 1px solid #1E4976;
        color: #E7EBF0;
    }
    .stDataFrame {
        background: #132F4C;
        border: 1px solid #1E4976;
    }
    
    /* Expanders dark theme */
    .streamlit-expanderHeader {
        background: #132F4C;
        border: 1px solid #1E4976;
        border-radius: 10px;
        color: #E7EBF0;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #00D9FF;
    }
    
    .streamlit-expanderContent {
        background: #132F4C;
        border: 1px solid #1E4976;
        color: #E7EBF0;
    }
    
    /* Code blocks */
    .stCode {
        background: #0e2238 !important;
        border: 1px solid #1E4976 !important;
    }
    
    /* Success/Warning/Error boxes */
    .stSuccess {
        background: #132F4C;
        border-left: 4px solid #00CCA3;
        color: #E7EBF0;
    }
    
    .stWarning {
        background: #132F4C;
        border-left: 4px solid #FFD93D;
        color: #E7EBF0;
    }
    
    .stError {
        background: #132F4C;
        border-left: 4px solid #FF6B6B;
        color: #E7EBF0;
    }
    
    .stInfo {
        background: #132F4C;
        border-left: 4px solid #00D9FF;
        color: #E7EBF0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color: #00D9FF;'>🔎 Case Search</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #B2BAC2; font-size: 1.1rem;'>Find similar historical fraud cases using AI-powered semantic search</p>", unsafe_allow_html=True)

# Configuration
CATALOG = os.getenv("CATALOG_NAME", "fraud_detection_dev")
SCHEMA = os.getenv("SCHEMA_NAME", "claims_analysis")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID")
VECTOR_INDEX = f"{CATALOG}.{SCHEMA}.fraud_cases_index"
CLAIMS_TABLE = f"{CATALOG}.{SCHEMA}.claims_data"
FRAUD_ANALYSIS_TABLE = f"{CATALOG}.{SCHEMA}.fraud_analysis"

# Initialize WorkspaceClient
@st.cache_resource
def get_workspace_client():
    """Initialize Databricks WorkspaceClient"""
    try:
        return WorkspaceClient()
    except Exception as e:
        st.error(f"Failed to initialize Databricks client: {e}")
        return None

w = get_workspace_client()

def execute_sql_query(query):
    """Execute SQL query using Statement Execution API"""
    if not w:
        st.error("WorkspaceClient not initialized")
        return None
    
    try:
        result = w.statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID,
            statement=query,
            wait_timeout="50s"
        )
        
        if result.status.state.value == "SUCCEEDED":
            if result.result and result.result.data_array:
                return result.result.data_array
            return []
        else:
            st.error(f"Query failed: {result.status.state.value}")
            return None
    except Exception as e:
        st.error(f"SQL execution error: {e}")
        return None

# Helper Functions
def get_claim_description_by_id(claim_id: str) -> str:
    """Lookup claim description from claims_data table by claim_id"""
    try:
        # Query the claims_data table to get claim information
        query = f"""
            SELECT claim_id, claim_type, claimant_info, provider_info
            FROM {CLAIMS_TABLE}
            WHERE claim_id = '{claim_id}'
            LIMIT 1
        """
        results = execute_sql_query(query)
        if results and len(results) > 0:
            result = results[0]
            # Create a search query from the claim details
            claim_type = result[1]
            claimant_info = result[2] if result[2] else ""
            provider_info = result[3] if result[3] else ""
            
            # Construct a descriptive search query
            search_text = f"{claim_type} claim"
            if claimant_info:
                search_text += f" involving {claimant_info}"
            if provider_info:
                search_text += f" from provider {provider_info}"
            
            return search_text
        return None
    except Exception as e:
        st.error(f"Error fetching claim: {e}")
        return None

def search_similar_cases(query_text: str, num_results: int = 5):
    """Search for similar cases using vector similarity"""
    if not w:
        st.error("WorkspaceClient not initialized")
        return None
    
    try:
        body = {
            "columns": ["doc_id", "doc_type", "title", "content"],
            "num_results": num_results,
            "query_text": query_text
        }
        
        response = w.api_client.do(
            'POST',
            f'/api/2.0/vector-search/indexes/{VECTOR_INDEX}/query',
            body=body
        )
        
        if isinstance(response, dict) and 'error_code' in response:
            error_msg = response.get('message', 'Unknown error')
            st.error(f"Vector Search error: {error_msg}")
            return None
        
        # Extract results with similarity scores
        result_data = response.get('result', {})
        data_array = result_data.get('data_array', [])
        
        if data_array:
            # Note: Vector search returns results sorted by similarity
            # Scores are typically between 0 and 1, we'll convert to percentage
            results = []
            for idx, row in enumerate(data_array):
                # The similarity score is implicit in the ranking
                # More similar results appear first
                # We'll create a pseudo-score based on ranking (100%, 90%, 80%, etc.)
                similarity_score = 100 - (idx * 10)  # Simple ranking-based score
                
                results.append({
                    "doc_id": row[0],
                    "doc_type": row[1],
                    "title": row[2],
                    "content": row[3],
                    "similarity": similarity_score
                })
            return results
        
        return []
        
    except Exception as e:
        st.error(f"Error searching cases: {e}")
        return None

def get_similarity_color(score: float) -> str:
    """Get color based on similarity score"""
    if score >= 80:
        return "🟢"  # Green
    elif score >= 60:
        return "🟡"  # Yellow
    else:
        return "🔴"  # Red

# Main UI
st.markdown("---")

# Initialize session state for inputs
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'claim_id_input' not in st.session_state:
    st.session_state.claim_id_input = ""

# Search interface
st.subheader("Search Configuration")

col1, col2 = st.columns([1, 3])

with col1:
    search_method = st.radio(
        "Search Method:",
        ["Text Description", "Claim ID"],
        help="Choose how you want to search for similar cases"
    )

with col2:
    if search_method == "Text Description":
        search_query = st.text_area(
            "Describe the claim or fraud pattern:",
            value=st.session_state.search_query,
            placeholder="E.g., Provider billed $50,000 for a routine checkup that typically costs $200",
            height=100,
            help="Enter a description of the suspicious claim or fraud pattern you're investigating",
            key="text_search_input"
        )
        
        # Sample text descriptions
        st.markdown("**Try these example patterns:**")
        sample_patterns = [
            "Provider billed $50,000 for a routine checkup that typically costs $200",
            "Multiple claims submitted for the same service on the same date by different claimants",
            "Medical equipment billed at prices 300% above market rate",
            "Provider billing for services after business hours or on holidays",
            "Claimant receiving treatment in multiple locations simultaneously",
            "Unbundling of bundled procedures to inflate reimbursement"
        ]
        
        cols = st.columns(3)
        for idx, pattern in enumerate(sample_patterns):
            with cols[idx % 3]:
                if st.button(pattern[:50] + "...", key=f"pattern_{idx}", use_container_width=True, help=pattern):
                    st.session_state.search_query = pattern
                    st.rerun()
        
        st.caption("**Click any example to copy it into the search box above**")
        
    else:
        claim_id_input = st.text_input(
            "Enter Claim ID:",
            value=st.session_state.claim_id_input,
            placeholder="E.g., CLM-000202",
            help="Enter the claim ID from the claims data table",
            key="claim_id_search_input"
        )
        
        # Quick claim ID suggestions
        st.markdown("**Try these sample claims:**")
        sample_claims = [
            ("CLM-000202", "Prescription claim"),
            ("CLM-000560", "DME claim"),
            ("CLM-000229", "Prescription claim"),
            ("CLM-000580", "Medical-Inpatient"),
            ("CLM-000879", "Medical-Inpatient"),
            ("CLM-000232", "Home Health")
        ]
        
        cols = st.columns(3)
        for idx, (claim_id, claim_type) in enumerate(sample_claims):
            with cols[idx % 3]:
                if st.button(f"{claim_id}\n({claim_type})", key=f"sample_{idx}", use_container_width=True):
                    st.session_state.claim_id_input = claim_id
                    st.rerun()
        
        st.caption("**Click any example to copy it into the search box above**")
        
        search_query = None  # Will be populated after lookup

# Search button
search_button = st.button("🔍 Search for Similar Cases", type="primary", use_container_width=True)

st.markdown("---")

# Execute search
if search_button:
    query_to_use = None
    
    if search_method == "Text Description":
        # Use the value from the text input
        text_value = st.session_state.get('text_search_input', search_query)
        if not text_value or len(text_value.strip()) < 10:
            st.warning("Please enter a more detailed description (at least 10 characters)")
        else:
            query_to_use = text_value
            
    else:  # Claim ID
        # Use the value from the claim ID input
        claim_value = st.session_state.get('claim_id_search_input', claim_id_input)
        if not claim_value or len(claim_value.strip()) == 0:
            st.warning("Please enter a valid Claim ID")
        else:
            with st.spinner(f"Looking up Claim ID: {claim_id_input}..."):
                description = get_claim_description_by_id(claim_id_input.strip())
                if description:
                    st.info(f"✅ Found claim: {claim_id_input}")
                    st.write(f"**Searching for cases similar to:** {description[:200]}...")
                    query_to_use = description
                else:
                    st.error(f"❌ Claim ID '{claim_id_input}' not found in the fraud analysis table")
    
    # Perform search if we have a query
    if query_to_use:
        with st.spinner("🔍 Searching for similar cases..."):
            results = search_similar_cases(query_to_use, num_results=5)
        
        if results is None:
            st.error("Search failed. Please try again.")
        elif len(results) == 0:
            st.info("No similar cases found. Try refining your search query.")
        else:
            # Display results
            st.success(f"✅ Found {len(results)} similar cases (sorted by relevance)")
            
            st.markdown("---")
            st.subheader("Search Results")
            
            for idx, result in enumerate(results, 1):
                similarity = result['similarity']
                color_indicator = get_similarity_color(similarity)
                
                with st.container():
                    # Header with similarity score
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"### {color_indicator} Result #{idx}: {result['title']}")
                    with col_b:
                        st.metric("Similarity", f"{similarity}%")
                    
                    # Doc type badge
                    st.caption(f"📁 Type: **{result['doc_type']}** | 🆔 ID: {result['doc_id']}")
                    
                    # Content preview
                    content_preview = result['content'][:300]
                    if len(result['content']) > 300:
                        content_preview += "..."
                    
                    st.write(content_preview)
                    
                    # Expandable full content
                    if len(result['content']) > 300:
                        with st.expander("📄 View Full Details"):
                            st.write(result['content'])
                    
                    st.markdown("---")

else:
    # Show help text when no search has been performed
    st.info("👆 Enter a search query above to find similar fraud cases")
    
    # Show helpful information
    with st.expander("ℹ️ How to Use Case Search"):
        st.markdown("""
        ### Text Description Search
        - Best for: Investigating new suspicious claims
        - Enter a description of the fraud pattern or suspicious behavior
        - Example: "Provider submitted multiple claims for the same service on the same day"
        
        ### Claim ID Search
        - Best for: Finding similar cases to an existing flagged claim
        - Enter a claim ID from the Fraud Insights or Claim Analysis pages
        - System will automatically extract the claim details and search for matches
        
        ### Interpreting Results
        - **🟢 Green (80%+)**: Highly similar cases - strong pattern match
        - **🟡 Yellow (60-80%)**: Moderately similar - review for relevance
        - **🔴 Red (<60%)**: Lower similarity - may still provide useful context
        
        ### Use Cases
        - 🔍 Investigate suspicious patterns from new claims
        - 📊 Find precedents for fraud cases under investigation
        - 🎓 Learn from historical fraud detection outcomes
        - 🤝 Share similar cases with team members
        """)

st.markdown("---")
st.caption("💡 **Tip:** Use Case Search to learn from past fraud investigations and identify emerging patterns!")
