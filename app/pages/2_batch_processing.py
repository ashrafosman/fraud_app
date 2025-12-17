"""
Batch Processing Page - Synchronous batch claim fraud analysis
"""

import streamlit as st
import pandas as pd
import os
import json
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

# Page configuration
st.set_page_config(
    page_title="Batch Processing | Fraud Detection",
    page_icon="⚡",
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
    .stTextInput > div > div, .stTextArea > div > div, .stSelectbox > div > div {
        background: #132F4C;
        border: 1px solid #1E4976;
        color: #E7EBF0;
    }
    .stDataFrame {
        background: #132F4C;
        border: 1px solid #1E4976;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color: #00D9FF;'>⚡ Batch Processing</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #B2BAC2; font-size: 1.1rem;'>Process multiple claims simultaneously with AI-powered fraud detection</p>", unsafe_allow_html=True)

# Configuration
CATALOG = os.getenv("CATALOG_NAME", "fraud_detection_dev")
SCHEMA = os.getenv("SCHEMA_NAME", "claims_analysis")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID")
VECTOR_INDEX = f"{CATALOG}.{SCHEMA}.fraud_cases_index"

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

# Sample claims data
SAMPLE_CLAIMS = {
    "Legitimate Office Visit": {
        "claim_id": "CLM-SAMPLE-001",
        "claim_text": "Medical claim for annual wellness visit. Preventive care exam with routine blood work. Member has consistent visit history with in-network provider. Diagnosis and procedure codes align correctly. Standard reimbursement request for $185.",
        "claim_amount": 185,
        "claim_type": "Medical-Outpatient"
    },
    "Upcoding Scheme": {
        "claim_id": "CLM-SAMPLE-002",
        "claim_text": "Provider billing CPT 99215 (complex visit) but documentation shows routine checkup (CPT 99213). Four similar high-complexity claims for same patient in 6 months. Diagnosis codes don't match procedure codes. Provider has pattern of upcoding. Billed $47,500 vs typical $185.",
        "claim_amount": 47500,
        "claim_type": "Medical-Outpatient"
    },
    "Phantom Billing": {
        "claim_id": "CLM-SAMPLE-003",
        "claim_text": "Provider billing for services member never received. Member confirmed they were out of state on date of service. Provider submitting claims for same patient on multiple dates when patient was traveling. Pattern of billing for non-existent appointments. Billed $12,000.",
        "claim_amount": 12000,
        "claim_type": "Medical-Outpatient"
    },
    "Prescription Fraud": {
        "claim_id": "CLM-SAMPLE-004",
        "claim_text": "Multiple high-cost controlled substance prescriptions filled at out-of-network pharmacy far from member's home. Same medications refilled early repeatedly. Prescriber has no prior relationship with patient. Pharmacy has pattern of early refills and doctor shopping indicators. Member has 8 different prescribers in 3 months. Billed $8,500.",
        "claim_amount": 8500,
        "claim_type": "Prescription"
    },
    "Legitimate Preventive": {
        "claim_id": "CLM-SAMPLE-005",
        "claim_text": "Routine dental cleaning and examination. Annual preventive care for established patient. No unusual procedures. Diagnosis codes normal. Member has regular 6-month dental visit history with this provider. Claim within expected cost range for preventive services. $0 copay.",
        "claim_amount": 0,
        "claim_type": "Dental"
    },
    "Unbundling Fraud": {
        "claim_id": "CLM-SAMPLE-006",
        "claim_text": "Lab billing each component test separately instead of bundled panel code. Same blood draw billed 15 times. Procedures that should be billed as single comprehensive metabolic panel unbundled into individual tests. Significantly inflated reimbursement at $15,000. Provider has pattern of unbundling across multiple claims.",
        "claim_amount": 15000,
        "claim_type": "Lab Services"
    }
}

# Helper Functions
def call_uc_function(function_name, *args, show_debug=False):
    """Call a Unity Catalog function using Statement Execution API"""
    try:
        # Escape single quotes in string arguments
        escaped_args = []
        for arg in args:
            if isinstance(arg, str):
                escaped_arg = arg.replace("'", "''")
                escaped_args.append(f"'{escaped_arg}'")
            else:
                escaped_args.append(str(arg))
        
        args_str = ', '.join(escaped_args)
        query = f"SELECT {CATALOG}.{SCHEMA}.{function_name}({args_str}) as result"
        
        result = w.statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID,
            statement=query,
            wait_timeout="50s"
        )
        
        if result.status.state.value == "SUCCEEDED":
            if result.result and result.result.data_array:
                data = result.result.data_array[0][0]
                
                if isinstance(data, str):
                    try:
                        parsed = json.loads(data)
                        return parsed
                    except:
                        return data
                elif isinstance(data, dict):
                    return data
                elif isinstance(data, (list, tuple)):
                    return data
                else:
                    return data
            return None
        else:
            if show_debug:
                st.error(f"Query failed: {result.status.state.value}")
            return None
    
    except Exception as e:
        if show_debug:
            st.error(f"Error calling UC function {function_name}: {e}")
        return None

def search_fraud_cases_vector(query: str, num_results: int = 3):
    """Search for similar fraud cases using vector search"""
    try:
        body = {
            "columns": ["doc_id", "doc_type", "title", "content"],
            "num_results": num_results,
            "query_text": query
        }
        
        response = w.api_client.do(
            'POST',
            f'/api/2.0/vector-search/indexes/{VECTOR_INDEX}/query',
            body=body
        )
        
        if isinstance(response, dict) and 'error_code' in response:
            return []
        
        data_array = response.get('result', {}).get('data_array', [])
        
        if data_array:
            formatted = []
            for row in data_array:
                formatted.append({
                    "doc_id": row[0],
                    "doc_type": row[1],
                    "title": row[2],
                    "content": row[3][:200]
                })
            return formatted
        return []
    except Exception as e:
        return []

def process_claim(claim_row, analysis_depth):
    """Process single claim based on depth selection"""
    result = {
        'claim_id': claim_row.get('claim_id', 'N/A'),
        'claim_text': claim_row.get('claim_text', '')[:100] + '...',
        'claim_amount': claim_row.get('claim_amount', 0)
    }
    
    # Always classify
    classify_result = call_uc_function("fraud_classify", claim_row['claim_text'])
    
    if classify_result:
        result['is_fraudulent'] = bool(classify_result.get('is_fraudulent', False))
        result['fraud_probability'] = float(classify_result.get('fraud_probability', 0.0))
        result['fraud_type'] = str(classify_result.get('fraud_type', 'None'))
        result['confidence'] = float(classify_result.get('confidence', 0.0))
        result['verdict'] = "FRAUD" if result['is_fraudulent'] else "LEGITIMATE"
    else:
        result['is_fraudulent'] = False
        result['fraud_probability'] = 0.0
        result['fraud_type'] = 'Error'
        result['confidence'] = 0.0
        result['verdict'] = "ERROR"
    
    # Standard: Add extraction
    if analysis_depth in ["Standard", "Deep"]:
        extract_result = call_uc_function("fraud_extract_indicators", claim_row['claim_text'])
        if extract_result:
            result['risk_score'] = extract_result.get('risk_score', 0)
            result['red_flags'] = extract_result.get('red_flags', [])
            result['urgency_level'] = extract_result.get('urgency_level', 'Low')
    
    # Deep: Add vector search
    if analysis_depth == "Deep":
        similar_cases = search_fraud_cases_vector(claim_row['claim_text'][:500], num_results=2)
        result['similar_cases_count'] = len(similar_cases)
        result['similar_cases'] = [case['title'] for case in similar_cases]
    
    return result

def save_results_to_table(results_df, table_name):
    """Save results DataFrame to Databricks table"""
    try:
        # Create table if not exists
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            claim_id STRING,
            claim_text STRING,
            claim_amount DOUBLE,
            is_fraudulent BOOLEAN,
            fraud_probability DOUBLE,
            fraud_type STRING,
            confidence DOUBLE,
            verdict STRING,
            processed_at TIMESTAMP
        )
        """
        
        w.statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID,
            statement=create_sql,
            wait_timeout="30s"
        )
        
        # Insert data (simplified - in production use proper DataFrame write)
        for _, row in results_df.iterrows():
            insert_sql = f"""
            INSERT INTO {table_name} VALUES (
                '{row['claim_id']}',
                '{row['claim_text'].replace("'", "''")}',
                {row.get('claim_amount', 0)},
                {str(row['is_fraudulent']).lower()},
                {row['fraud_probability']},
                '{row['fraud_type']}',
                {row['confidence']},
                '{row['verdict']}',
                current_timestamp()
            )
            """
            
            w.statement_execution.execute_statement(
                warehouse_id=WAREHOUSE_ID,
                statement=insert_sql,
                wait_timeout="30s"
            )
        
        return True
    except Exception as e:
        st.error(f"Error saving to table: {e}")
        return False

# Initialize session state
if 'batch_claims' not in st.session_state:
    st.session_state.batch_claims = None
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'selected_samples' not in st.session_state:
    st.session_state.selected_samples = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# Main UI
st.markdown("---")

# Input Section
st.subheader("📥 Step 1: Load Claims")

tab1, tab2 = st.tabs(["📁 Upload File", "🎯 Select Samples"])

with tab1:
    st.markdown("Upload a CSV or Excel file with claim data")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="File must contain columns: claim_id, claim_text. Optional: claim_amount, claim_type"
    )
    
    if uploaded_file:
        try:
            # Read file based on extension
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            
            # Validate required columns
            required_cols = ['claim_id', 'claim_text']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("Required columns: claim_id, claim_text")
            else:
                st.success(f"✅ File loaded successfully: {len(df)} claims")
                
                # Show preview
                st.markdown("**Preview (first 5 rows):**")
                st.dataframe(df.head(), use_container_width=True)
                
                # Store in session state only if it's a new file or different data
                # Check if this is actually new data to avoid resetting results
                is_new_data = (
                    st.session_state.batch_claims is None or 
                    len(st.session_state.batch_claims) != len(df) or
                    not st.session_state.batch_claims.equals(df)
                )
                
                if is_new_data:
                    st.session_state.batch_claims = df
                    st.session_state.processing_complete = False
                    st.session_state.batch_results = None
                
        except Exception as e:
            st.error(f"Error reading file: {e}")

with tab2:
    st.markdown("Select up to 5 sample claims for demonstration")
    
    # Display sample claim buttons in grid
    cols = st.columns(3)
    
    for idx, (name, claim_data) in enumerate(SAMPLE_CLAIMS.items()):
        col_idx = idx % 3
        with cols[col_idx]:
            # Create checkbox for selection
            is_selected = idx in st.session_state.selected_samples
            
            if st.checkbox(
                name,
                value=is_selected,
                key=f"sample_check_{idx}",
                disabled=len(st.session_state.selected_samples) >= 5 and not is_selected
            ):
                if idx not in st.session_state.selected_samples:
                    st.session_state.selected_samples.append(idx)
            else:
                if idx in st.session_state.selected_samples:
                    st.session_state.selected_samples.remove(idx)
            
            # Show claim info
            st.caption(f"${claim_data['claim_amount']:,.0f} | {claim_data['claim_type']}")
    
    # Show selection counter
    st.markdown(f"**Selected: {len(st.session_state.selected_samples)}/5 claims**")
    
    # Load selected claims button
    if st.button("📋 Load Selected Claims", disabled=len(st.session_state.selected_samples) == 0):
        selected_claims = []
        for idx in st.session_state.selected_samples:
            claim_name = list(SAMPLE_CLAIMS.keys())[idx]
            claim_data = SAMPLE_CLAIMS[claim_name]
            selected_claims.append(claim_data)
        
        df = pd.DataFrame(selected_claims)
        
        # Only reset if it's actually new data
        st.session_state.batch_claims = df
        st.session_state.processing_complete = False
        st.session_state.batch_results = None
        
        st.success(f"✅ Loaded {len(df)} sample claims")
        st.rerun()

# Configuration Section
if st.session_state.batch_claims is not None and not st.session_state.batch_claims.empty:
    st.markdown("---")
    st.subheader("⚙️ Step 2: Configure Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        analysis_depth = st.radio(
            "Select analysis depth:",
            ["Quick: Classify only", "Standard: Classify + Extract", "Deep: All Tools"],
            help="""
            - Quick: Fast classification only (~2s per claim)
            - Standard: Classification + detailed indicators (~4s per claim)
            - Deep: Full analysis with similar case search (~6s per claim)
            """
        )
    
    with col2:
        st.metric("Claims to Process", len(st.session_state.batch_claims))
        
        # Estimate processing time
        depth_mapping = {
            "Quick: Classify only": 2,
            "Standard: Classify + Extract": 4,
            "Deep: All Tools": 6
        }
        depth_key = analysis_depth
        est_time = len(st.session_state.batch_claims) * depth_mapping[depth_key]
        st.metric("Est. Time", f"{est_time}s")
    
    # Processing Section
    st.markdown("---")
    st.subheader("🚀 Step 3: Process Claims")
    
    if st.button("▶️ Start Batch Analysis", type="primary", use_container_width=True):
        # Extract depth value
        depth_value = analysis_depth.split(":")[0]  # "Quick", "Standard", or "Deep"
        
        st.markdown("### Processing in Progress...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        live_results_placeholder = st.empty()
        
        results = []
        total_claims = len(st.session_state.batch_claims)
        start_time = time.time()
        
        for idx, row in st.session_state.batch_claims.iterrows():
            status_text.text(f"Processing {idx+1}/{total_claims}: {row['claim_id']}")
            
            # Process claim
            result = process_claim(row, depth_value)
            results.append(result)
            
            # Update progress
            progress_bar.progress((idx + 1) / total_claims)
            
            # Show live preview
            if len(results) > 0:
                preview_df = pd.DataFrame(results)
                live_results_placeholder.dataframe(
                    preview_df[['claim_id', 'verdict', 'fraud_probability', 'fraud_type']].tail(5),
                    use_container_width=True
                )
        
        elapsed_time = time.time() - start_time
        
        # Store results in session state
        st.session_state.batch_results = pd.DataFrame(results)
        st.session_state.processing_complete = True
        st.session_state.processing_time = elapsed_time
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        live_results_placeholder.empty()
        
        st.success(f"✅ Processing complete! Analyzed {total_claims} claims in {elapsed_time:.1f} seconds")
        
        # Force rerun to show results section
        st.rerun()

# Results Section - Shows independently of file upload status
if st.session_state.processing_complete and st.session_state.batch_results is not None:
    st.markdown("---")
    st.subheader("📊 Step 4: View Results")
    
    results_df = st.session_state.batch_results
    
    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_claims = len(results_df)
    # Handle both boolean and string representations
    fraud_claims = len(results_df[
        (results_df['is_fraudulent'] == True) | 
        (results_df['is_fraudulent'] == 'true') |
        (results_df['verdict'] == 'FRAUD')
    ])
    legitimate_claims = total_claims - fraud_claims
    fraud_percentage = (fraud_claims / total_claims * 100) if total_claims > 0 else 0
    
    with col1:
        st.metric("Total Processed", total_claims)
    with col2:
        st.metric("Fraudulent Claims", fraud_claims, delta=f"{fraud_percentage:.1f}%")
    with col3:
        st.metric("Legitimate Claims", legitimate_claims)
    with col4:
        processing_time = st.session_state.get('processing_time', 0)
        st.metric("Processing Time", f"{processing_time:.1f}s")
    
    st.markdown("---")
    
    # Results Table with Filters
    st.markdown("### Detailed Results")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        verdict_filter = st.selectbox(
            "Filter by verdict:",
            ["All", "FRAUD", "LEGITIMATE"]
        )
    
    # Apply filter
    if verdict_filter == "All":
        filtered_df = results_df
    else:
        filtered_df = results_df[results_df['verdict'] == verdict_filter]
    
    st.markdown(f"**Showing {len(filtered_df)} of {len(results_df)} claims**")
    
    # Display table with custom formatting
    display_df = filtered_df.copy()
    
    st.dataframe(
        display_df,
        column_config={
            "claim_id": st.column_config.TextColumn("Claim ID", width="medium"),
            "fraud_probability": st.column_config.ProgressColumn(
                "Fraud Risk",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "verdict": st.column_config.TextColumn("Verdict", width="small"),
            "fraud_type": st.column_config.TextColumn("Fraud Type", width="medium"),
            "confidence": st.column_config.ProgressColumn(
                "Confidence",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "claim_amount": st.column_config.NumberColumn(
                "Amount",
                format="$%.0f",
            ),
        },
        use_container_width=True,
        hide_index=True
    )
    
    # Export Options
    st.markdown("---")
    st.markdown("### 📤 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download CSV
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"fraud_batch_results_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Save to Databricks table
        if st.button("💾 Save to Databricks Table", use_container_width=True):
            table_name = f"{CATALOG}.{SCHEMA}.batch_results"
            
            with st.spinner(f"Saving to {table_name}..."):
                success = save_results_to_table(results_df, table_name)
                
                if success:
                    st.success(f"✅ Results saved to table: {table_name}")
                else:
                    st.error("❌ Failed to save results")

# Help Section
st.markdown("---")
with st.expander("ℹ️ How to Use Batch Processing"):
    st.markdown("""
    ### Upload File Method
    1. Prepare a CSV or Excel file with at least these columns:
       - `claim_id` - Unique identifier for each claim
       - `claim_text` - Full text description of the claim
       - `claim_amount` (optional) - Dollar amount
       - `claim_type` (optional) - Type of claim
    
    2. Upload the file using the file uploader
    3. Preview the data to ensure it loaded correctly
    
    ### Sample Claims Method
    1. Check the boxes next to sample claims (max 5)
    2. Click "Load Selected Claims"
    3. Sample claims are pre-configured for demonstration
    
    ### Analysis Depth Options
    - **Quick**: Fast classification only - good for large batches
    - **Standard**: Adds detailed fraud indicator extraction
    - **Deep**: Full analysis including similar case search
    
    ### Results
    - View summary metrics at the top
    - Filter and sort the detailed results table
    - Download results as CSV or save to Databricks table
    """)

st.markdown("---")
st.caption("⚡ Batch Processing | Built with Unity Catalog AI Functions | Databricks Apps")
