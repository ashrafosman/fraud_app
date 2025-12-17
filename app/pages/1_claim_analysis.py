"""
Claim Analysis Page - AI-Powered Fraud Detection with LangGraph Agent
Based on databricks-ai-ticket-vectorsearch pattern
"""

import streamlit as st
import os
import json
import time
from databricks.sdk import WorkspaceClient

# Page config
st.set_page_config(page_title="Claim Analysis", page_icon="📊", layout="wide")

# Dark tech theme CSS
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
    
    .modern-title {
        background: linear-gradient(90deg, #00D9FF 0%, #00CCA3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 10px;
    }
    
    .info-box {
        background: linear-gradient(135deg, #132F4C 0%, #0e2238 100%);
        border: 1px solid #1E4976;
        color: #E7EBF0;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
    }
    
    .sample-card {
        background: #132F4C;
        border: 1px solid #1E4976;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #00D9FF;
        transition: all 0.3s ease;
    }
    
    .sample-card:hover {
        border-color: #00D9FF;
        box-shadow: 0 8px 25px rgba(0,217,255,0.2);
        transform: translateX(5px);
    }
    
    .tool-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 5px;
    }
    
    .badge-classify {
        background: #00D9FF;
        color: #0A1929;
    }
    
    .badge-extract {
        background: #00CCA3;
        color: #0A1929;
    }
    
    .badge-search {
        background: #1E4976;
        color: #00D9FF;
    }
    
    .badge-explain {
        background: #FFD93D;
        color: #0A1929;
    }
    
    /* Text inputs dark theme */
    .stTextInput > div > div, .stTextArea > div > div {
        background: #132F4C;
        border: 1px solid #1E4976;
        color: #E7EBF0;
    }
    
    .stSelectbox > div > div {
        background: #132F4C;
        border: 1px solid #1E4976;
        color: #E7EBF0;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #00D9FF !important;
    }
    
    p, li, span, label {
        color: #E7EBF0 !important;
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
    
    code {
        background: #0e2238 !important;
        color: #00D9FF !important;
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
    
    /* Dropdown options styling */
    .stSelectbox > div > div > div {
        background: #132F4C;
        color: #E7EBF0;
    }
    
    [data-baseweb="select"] > div {
        background: #132F4C;
        color: #E7EBF0;
    }
    
    [role="listbox"] {
        background: #132F4C !important;
        border: 1px solid #1E4976 !important;
    }
    
    [role="option"] {
        background: #132F4C !important;
        color: #E7EBF0 !important;
    }
    
    [role="option"]:hover {
        background: #1E4976 !important;
        color: #00D9FF !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='modern-title'>📊 AI Claim Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 1.2rem; color: #B2BAC2; margin-top: -10px;'>Intelligent fraud detection powered by LangGraph AI Agent</p>", unsafe_allow_html=True)

# Read configuration from environment (set in app.yaml)
CATALOG = os.getenv("CATALOG_NAME", "fraud_detection_dev")
SCHEMA = os.getenv("SCHEMA_NAME", "claims_analysis")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "148ccb90800933a1")
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
VECTOR_INDEX = f"{CATALOG}.{SCHEMA}.fraud_cases_index"

# Initialize Databricks client
@st.cache_resource
def get_workspace_client():
    """Initialize Databricks WorkspaceClient (automatically authenticated in Databricks Apps)"""
    try:
        return WorkspaceClient()
    except Exception as e:
        st.error(f"Failed to initialize Databricks client: {e}")
        return None

w = get_workspace_client()

def call_uc_function(function_name, *args, timeout=50, show_debug=False):
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
        
        if show_debug:
            st.info(f"🔍 Executing: {function_name}(...) on warehouse {WAREHOUSE_ID}")
        
        result = w.statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID,
            statement=query,
            wait_timeout="50s"
        )
        
        if result.status.state.value == "SUCCEEDED":
            if result.result and result.result.data_array:
                data = result.result.data_array[0][0]
                
                if isinstance(data, str):
                    import json
                    try:
                        parsed = json.loads(data)
                        return parsed
                    except:
                        return data
                elif isinstance(data, dict):
                    return data
                elif isinstance(data, (list, tuple)):
                    # For STRUCT types returned as arrays
                    if function_name == "fraud_generate_explanation" and len(data) >= 3:
                        return {
                            'summary': data[0],
                            'key_findings': data[1] if data[1] else [],
                            'recommendations': data[2] if data[2] else []
                        }
                    return data
                else:
                    return data
            return None
        else:
            if show_debug:
                st.error(f"Query failed: {result.status.state.value}")
                if result.status.error:
                    st.error(f"Error: {result.status.error.message}")
            return None
    
    except Exception as e:
        if show_debug:
            st.error(f"Error calling UC function {function_name}: {e}")
        return None

# ===== LANGCHAIN TOOLS FOR LANGRAPH AGENT =====
try:
    from langchain_core.tools import Tool, StructuredTool
    from pydantic import BaseModel, Field
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import SystemMessage
    from databricks_langchain import ChatDatabricks
    
    LANGCHAIN_AVAILABLE = True
    
    # Tool input schemas
    class ClassifyClaimInput(BaseModel):
        claim_text: str = Field(description="The insurance claim text to classify for fraud")
    
    class ExtractIndicatorsInput(BaseModel):
        claim_text: str = Field(description="The claim text to extract fraud indicators from")
    
    class SearchFraudPatternsInput(BaseModel):
        query: str = Field(description="The search query to find relevant fraud patterns")
    
    class GenerateExplanationInput(BaseModel):
        claim_text: str = Field(description="The claim text to explain")
        is_fraudulent: bool = Field(description="Whether the claim is fraudulent (from classification)")
        fraud_type: str = Field(description="Type of fraud detected (from classification)", default="none")
    
    # Tool wrapper functions
    def classify_claim_wrapper(claim_text: str) -> str:
        """Classifies a claim as fraudulent or legitimate"""
        result = call_uc_function("fraud_classify", claim_text, show_debug=False)
        import json
        return json.dumps(result, indent=2) if result else json.dumps({"error": "Classification failed"})
    
    def extract_indicators_wrapper(claim_text: str) -> str:
        """Extracts fraud indicators from a claim"""
        result = call_uc_function("fraud_extract_indicators", claim_text, show_debug=False)
        import json
        return json.dumps(result, indent=2) if result else json.dumps({"error": "Extraction failed"})
    
    def search_fraud_patterns_wrapper(query: str) -> str:
        """Searches the fraud knowledge base for relevant patterns"""
        import json
        try:
            if not w:
                return json.dumps({"error": "WorkspaceClient not initialized"})
            
            body = {
                "columns": ["doc_id", "doc_type", "title", "content"],
                "num_results": 3,
                "query_text": query
            }
            
            response = w.api_client.do(
                'POST',
                f'/api/2.0/vector-search/indexes/{VECTOR_INDEX}/query',
                body=body
            )
            
            if isinstance(response, dict) and 'error_code' in response:
                error_msg = response.get('message', 'Unknown error')
                return json.dumps({"error": f"Vector Search error: {error_msg}"})
            
            data_array = response.get('result', {}).get('data_array', [])
            
            if data_array:
                formatted = []
                for row in data_array:
                    formatted.append({
                        "doc_id": row[0],
                        "doc_type": row[1],
                        "title": row[2],
                        "content": row[3][:500]  # Truncate for agent
                    })
                return json.dumps(formatted, indent=2)
            return json.dumps([])
        except Exception as e:
            return json.dumps({"error": f"Search failed: {str(e)}"})
    
    def generate_explanation_wrapper(claim_text: str, is_fraudulent: bool, fraud_type: str = "none") -> str:
        """Generates comprehensive fraud explanation with risk factors and recommendations"""
        result = call_uc_function("fraud_generate_explanation", claim_text, is_fraudulent, fraud_type, show_debug=False)
        import json
        return json.dumps(result, indent=2) if result else json.dumps({"error": "Explanation generation failed"})
    
    # Create LangChain Tools
    classify_tool = Tool(
        name="classify_claim",
        description="Classifies a healthcare claim as fraudulent or legitimate. Use this FIRST to understand fraud risk. Returns JSON with is_fraudulent, fraud_probability, fraud_type, confidence.",
        func=classify_claim_wrapper,
        args_schema=ClassifyClaimInput
    )
    
    extract_tool = Tool(
        name="extract_indicators",
        description="Extracts fraud indicators from claim including risk score, red flags, anomaly indicators, urgency level, and financial impact. Use after classification to get detailed analysis. Returns JSON with structured indicators.",
        func=extract_indicators_wrapper,
        args_schema=ExtractIndicatorsInput
    )
    
    search_tool = Tool(
        name="search_fraud_patterns",
        description="Searches the fraud knowledge base for relevant patterns, schemes, and documentation using semantic search. Use to find similar fraud cases or detection techniques. Returns JSON array with title, content, fraud_type for top matches.",
        func=search_fraud_patterns_wrapper,
        args_schema=SearchFraudPatternsInput
    )
    
    explain_tool = StructuredTool.from_function(
        func=generate_explanation_wrapper,
        name="generate_explanation",
        description="Generates comprehensive fraud explanation with summary, risk factors, recommendations. REQUIRES results from classify_claim first. Pass claim_text, is_fraudulent (true/false), and fraud_type from classification. Returns JSON with detailed explanation.",
        args_schema=GenerateExplanationInput
    )
    
    # LangGraph Agent creation
    @st.cache_resource
    def create_langraph_agent():
        """Create the LangGraph ReAct agent with all tools"""
        try:
            # Use Claude Sonnet 4.5 for EXCELLENT function calling support
            agent_endpoint = os.getenv("LLM_ENDPOINT", "databricks-claude-sonnet-4-5")
            
            # Initialize LLM
            llm = ChatDatabricks(
                endpoint=agent_endpoint,
                temperature=0.1,  # Low temp for reliable tool calls
                max_tokens=2000
            )
            
            # Create agent with all tools
            tools_list = [classify_tool, extract_tool, search_tool, explain_tool]
            
            # CRITICAL: Bind tools to LLM for consistent JSON format
            llm_with_tools = llm.bind_tools(tools_list)
            
            agent = create_react_agent(
                model=llm_with_tools,
                tools=tools_list
            )
            
            return agent
        except Exception as e:
            st.error(f"Error creating agent: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None
    
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    st.warning(f"LangChain/LangGraph not available: {e}")

# Sample claims for testing (Healthcare Payer scenarios)
SAMPLE_CLAIMS = {
    "Legitimate Office Visit": """Medical claim #CLM-2024-001
Member ID: MEM-456789
Provider: Dr. Sarah Chen, Internal Medicine
Date of Service: 2024-12-10
Billed Amount: $185
Description: Annual wellness visit for established patient. Preventive care exam with routine blood work. Member has consistent visit history with this in-network provider. Diagnosis codes and procedure codes align correctly. Standard reimbursement request.""",

    "Upcoding Scheme": """Medical claim #CLM-2024-045
Member ID: MEM-123456
Provider: QuickCare Medical Center (Out-of-Network)
Date of Service: 2024-12-12
Billed Amount: $47,500
Description: Provider billing for complex surgical procedures but documentation shows only routine office visit. Four similar high-complexity claims for same patient in 6 months. Diagnosis codes (routine checkup) don't match procedure codes (major surgery). Provider has pattern of upcoding across multiple patients. Medical necessity not established.""",

    "Phantom Billing": """Medical claim #CLM-2024-089
Member ID: MEM-789012
Provider: Metro Health Services
Date of Service: 2024-12-08
Billed Amount: $12,000
Description: Provider billing for services member never received. Member confirmed they were out of state on date of service. Provider submitting claims for same patient on multiple dates when patient was traveling. Pattern of billing for non-existent appointments. Provider address flagged as residential location.""",

    "Prescription Drug Diversion": """Pharmacy claim #CLM-2024-112
Member ID: MEM-345678
Provider: Valley Pharmacy (Out-of-Network)
Date of Service: 2024-12-05
Billed Amount: $8,500
Description: Multiple high-cost controlled substance prescriptions filled at out-of-network pharmacy far from member's home. Same medications refilled early repeatedly. Prescriber has no prior relationship with patient. Pharmacy has pattern of early refills and doctor shopping indicators. Member has 8 different prescribers in 3 months.""",

    "Legitimate Preventive Care": """Medical claim #CLM-2024-067
Member ID: MEM-234567
Provider: Healthy Smiles Dental (In-Network)
Date of Service: 2024-12-11
Billed Amount: $0 (Preventive - 100% coverage)
Description: Routine dental cleaning and examination. Annual preventive care for established patient. No unusual procedures. Diagnosis codes normal. Member has regular 6-month dental visit history with this provider. Claim within expected cost range for preventive services.""",

    "Unbundling Fraud": """Medical claim #CLM-2024-134
Member ID: MEM-567890
Provider: Advanced Diagnostics Lab
Date of Service: 2024-12-13
Billed Amount: $15,000
Description: Lab billing each component test separately instead of bundled panel code. Same blood draw billed 15 times. Procedures that should be billed as single comprehensive metabolic panel unbundled into individual tests. Significantly inflated reimbursement. Provider has pattern of unbundling across multiple claims.""",
}

# Main UI
st.markdown("<br>", unsafe_allow_html=True)

if not LANGCHAIN_AVAILABLE:
    st.markdown("""
    <div style='background: #FF6B6B; color: white; padding: 20px; border-radius: 15px; margin: 20px 0;'>
        <h3 style='margin: 0 0 10px 0;'>❌ LangChain/LangGraph Not Available</h3>
        <p style='margin: 0;'>Please install required packages:</p>
        <code style='background: rgba(0,0,0,0.2); padding: 10px; display: block; margin-top: 10px; border-radius: 5px;'>
        pip install langgraph>=1.0.0 langchain>=0.3.0 langchain-core>=0.3.0 databricks-langchain
        </code>
    </div>
    """, unsafe_allow_html=True)
else:
    # Agent Info Box
    st.markdown("""
    <div class='info-box'>
        <h3 style='margin: 0 0 15px 0; color: #00D9FF;'>🧠 LangGraph ReAct Agent</h3>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;'>
            <div>
                <div style='font-size: 1.5rem; margin-bottom: 5px;'>🧠</div>
                <b style='color: #E7EBF0;'>Think</b><br/>
                <small style='opacity: 0.8; color: #B2BAC2;'>Analyze which tools to use</small>
            </div>
            <div>
                <div style='font-size: 1.5rem; margin-bottom: 5px;'>🔧</div>
                <b style='color: #E7EBF0;'>Act</b><br/>
                <small style='opacity: 0.8; color: #B2BAC2;'>Execute fraud detection tools</small>
            </div>
            <div>
                <div style='font-size: 1.5rem; margin-bottom: 5px;'>🔄</div>
                <b style='color: #E7EBF0;'>Observe</b><br/>
                <small style='opacity: 0.8; color: #B2BAC2;'>Review results and iterate</small>
            </div>
            <div>
                <div style='font-size: 1.5rem; margin-bottom: 5px;'>🎯</div>
                <b style='color: #E7EBF0;'>Adapt</b><br/>
                <small style='opacity: 0.8; color: #B2BAC2;'>Adjust strategy dynamically</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Claim input section
    st.markdown("<h3 style='color: #667eea;'>📝 Claim Input</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        sample_choice = st.selectbox(
            "🔖 Select a sample claim or enter your own:", 
            ["Custom"] + list(SAMPLE_CLAIMS.keys()),
            help="Choose from pre-loaded healthcare claim examples or create your own"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if sample_choice != "Custom":
            # Show badge for fraud status
            is_fraud = sample_choice in ["Upcoding Scheme", "Phantom Billing", "Prescription Drug Diversion", "Unbundling Fraud"]
            badge_color = "#FF6B6B" if is_fraud else "#4ECDC4"
            badge_text = "🚨 FRAUD" if is_fraud else "✅ LEGITIMATE"
            st.markdown(f"""
            <div style='background: {badge_color}; color: white; padding: 10px; border-radius: 10px; text-align: center; font-weight: 600; font-size: 0.9rem;'>
                {badge_text}
            </div>
            """, unsafe_allow_html=True)
    
    if sample_choice == "Custom":
        claim_text = st.text_area(
            "📄 Enter claim details:", 
            height=220, 
            value="",
            placeholder="Enter claim ID, member ID, provider, date of service, billed amount, and description...\n\nExample:\nMedical claim #CLM-2024-001\nMember ID: MEM-456789\nProvider: Dr. Sarah Chen\nDate of Service: 2024-12-10\nBilled Amount: $185\nDescription: Annual wellness visit..."
        )
    else:
        claim_text = st.text_area(
            "📄 Claim details:", 
            height=220, 
            value=SAMPLE_CLAIMS[sample_choice]
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        analyze_btn = st.button("🚀 Analyze with AI Agent", use_container_width=True)
    with col2:
        if claim_text:
            word_count = len(claim_text.split())
            st.markdown(f"""
            <div style='background: #f8f9fa; padding: 12px; border-radius: 8px; text-align: center; margin-top: 5px;'>
                <div style='color: #667eea; font-weight: 600;'>{word_count}</div>
                <div style='font-size: 0.8rem; color: #999;'>words</div>
            </div>
            """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style='background: #132F4C; border: 1px solid #1E4976; padding: 12px; border-radius: 8px; margin-top: 5px;'>
            <div style='font-size: 0.85rem; color: #E7EBF0;'>
                <b>Available Tools:</b>
                <span class='tool-badge badge-classify'>🎯 Classify</span>
                <span class='tool-badge badge-extract'>📊 Extract</span>
                <span class='tool-badge badge-search'>🔍 Search</span>
                <span class='tool-badge badge-explain'>💡 Explain</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if analyze_btn:
        if not claim_text.strip():
            st.markdown("""
            <div style='background: #FFE66D; color: #333; padding: 15px; border-radius: 10px; margin: 20px 0;'>
                <b>⚠️ Please enter claim details</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<br><hr style='border: 1px solid #eee; margin: 30px 0;'><br>", unsafe_allow_html=True)
            st.markdown("""
            <div style='text-align: center;'>
                <h2 style='color: #667eea; margin-bottom: 10px;'>🤖 AI Agent Analysis</h2>
                <p style='color: #666;'>Watch the AI agent reason through the fraud detection process</p>
            </div>
            <br>
            """, unsafe_allow_html=True)
            
            total_start = time.time()
            
            # Create agent
            agent = create_langraph_agent()
            
            if not agent:
                st.markdown("""
                <div style='background: #FF6B6B; color: white; padding: 20px; border-radius: 15px;'>
                    <h3 style='margin: 0;'>❌ Failed to create LangGraph agent</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                # System prompt
                system_prompt = """You are an expert healthcare fraud detection analyst for insurance payers (Humana, UHG, Cigna, etc.). Your job is to analyze claims and detect fraud.

You have access to these tools:
1. classify_claim - Determines if claim is fraudulent (returns is_fraudulent, fraud_type, etc). Use this FIRST.
2. extract_indicators - Extracts detailed fraud indicators. Use after classification.
3. search_fraud_patterns - Searches fraud knowledge base. Use for most claims.
4. generate_explanation - Creates comprehensive explanation. MUST pass is_fraudulent and fraud_type from classify_claim results.

IMPORTANT: You MUST use the tools by calling them properly. After using tools, provide a final analysis.

Analysis strategy:
- Start with classify_claim (get is_fraudulent and fraud_type)
- Then use extract_indicators
- Use search_fraud_patterns to find similar fraud cases
- Use generate_explanation with the is_fraudulent and fraud_type from step 1
- After gathering information, provide your final fraud assessment

Be thorough but efficient."""
                
                # Container for agent reasoning
                reasoning_container = st.container()
                
                with reasoning_container:
                    st.markdown("#### 🧠 Agent Reasoning Process")
                    
                    # Show agent thinking
                    with st.spinner("🤔 Agent is analyzing the claim..."):
                        try:
                            # Invoke agent with system message
                            result = agent.invoke({
                                "messages": [
                                    SystemMessage(content=system_prompt),
                                    ("user", f"Analyze this healthcare claim for fraud and provide a comprehensive assessment: {claim_text}")
                                ]
                            })
                            
                            elapsed_time = (time.time() - total_start) * 1000
                            
                            # Parse messages to show reasoning
                            messages = result.get('messages', [])
                            
                            st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #00CCA3 0%, #00D9FF 100%); color: #0A1929; padding: 20px; border-radius: 15px; text-align: center; margin: 20px 0; box-shadow: 0 8px 25px rgba(0,204,163,0.3);'>
                                <h3 style='margin: 0; color: #0A1929;'>✅ Analysis Complete!</h3>
                                <p style='margin: 10px 0 0 0; font-size: 1.1rem; color: #0A1929;'>Processed in <b>{elapsed_time:.0f}ms</b></p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Show tool calls and reasoning
                            tool_calls = []
                            agent_thoughts = []
                            agent_response = None
                            
                            for msg in messages:
                                msg_type = getattr(msg, 'type', None) or type(msg).__name__.lower()
                                
                                if 'ai' in msg_type:
                                    # AI message (thought or final response)
                                    content = getattr(msg, 'content', '')
                                    tool_calls_in_msg = getattr(msg, 'tool_calls', [])
                                    
                                    if tool_calls_in_msg:
                                        # This is a thought with tool calls
                                        for tc in tool_calls_in_msg:
                                            tool_name = tc.get('name', 'unknown')
                                            tool_args = tc.get('args', {})
                                            tool_calls.append({
                                                'name': tool_name,
                                                'args': tool_args
                                            })
                                    elif content:
                                        # This is reasoning or final answer
                                        if not agent_response:  # First AI message with content is likely the final answer
                                            agent_response = content
                                        else:
                                            agent_thoughts.append(content)
                                
                                elif 'tool' in msg_type:
                                    # Tool response
                                    tool_name = getattr(msg, 'name', 'unknown')
                                    tool_content = getattr(msg, 'content', '')
                                    
                                    # Find matching tool call
                                    for tc in tool_calls:
                                        if tc['name'] == tool_name and 'result' not in tc:
                                            tc['result'] = tool_content
                                            break
                            
                            # Display tool calls in expandable sections
                            if tool_calls:
                                st.markdown("""
                                <div style='background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); padding: 20px; border-radius: 15px; margin: 20px 0;'>
                                    <h3 style='color: #667eea; margin: 0 0 10px 0;'>🔧 Agent Tools Execution</h3>
                                    <p style='color: #666; margin: 0;'>The AI agent intelligently used <b style='color: #667eea;'>{}</b> out of 4 available tools</p>
                                </div>
                                """.format(len(tool_calls)), unsafe_allow_html=True)
                                
                                for i, tc in enumerate(tool_calls, 1):
                                    tool_name = tc['name']
                                    tool_args = tc.get('args', {})
                                    tool_result = tc.get('result', 'No result')
                                    
                                    # Icon and color based on tool
                                    if "classify" in tool_name:
                                        icon = "🎯"
                                        color = "#667eea"
                                        badge_class = "badge-classify"
                                    elif "extract" in tool_name:
                                        icon = "📊"
                                        color = "#4ECDC4"
                                        badge_class = "badge-extract"
                                    elif "search" in tool_name:
                                        icon = "🔍"
                                        color = "#F093FB"
                                        badge_class = "badge-search"
                                    else:
                                        icon = "💡"
                                        color = "#FFE66D"
                                        badge_class = "badge-explain"
                                    
                                    with st.expander(f"{icon} **Tool {i}: {tool_name}**", expanded=(i <= 2)):
                                        st.markdown(f"""
                                        <div style='background: {color}15; padding: 15px; border-radius: 10px; border-left: 4px solid {color}; margin-bottom: 15px;'>
                                            <h4 style='color: {color}; margin: 0 0 10px 0;'>{icon} {tool_name}</h4>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        st.markdown("**📥 Input Parameters:**")
                                        st.json(tool_args)
                                        
                                        st.markdown("**📤 Output Result:**")
                                        try:
                                            result_json = json.loads(tool_result)
                                            st.json(result_json)
                                        except:
                                            st.text(tool_result[:500] + "..." if len(tool_result) > 500 else tool_result)
                            else:
                                st.markdown("""
                                <div style='background: #FFE66D; color: #333; padding: 15px; border-radius: 10px;'>
                                    <b>⚠️ No tool calls detected in agent execution</b>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Display final response
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("""
                            <div style='background: linear-gradient(135deg, #132F4C 0%, #0e2238 100%); border: 1px solid #00D9FF; padding: 25px; border-radius: 15px; color: #E7EBF0; margin: 20px 0;'>
                                <h3 style='margin: 0 0 15px 0; color: #00D9FF;'>💡 Final Fraud Assessment</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            if agent_response:
                                st.markdown(f"""
                                <div style='background: #132F4C; border: 1px solid #1E4976; padding: 25px; border-radius: 15px; margin: 20px 0; color: #E7EBF0;'>
                                    {agent_response}
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div style='background: rgba(0,204,163,0.1); border: 1px solid #00CCA3; padding: 20px; border-radius: 10px; color: #00CCA3;'>
                                    <b>✓ Agent completed analysis. Check tool outputs above for details.</b>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Performance metrics
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("<h4 style='color: #00D9FF; text-align: center;'>📊 Performance Metrics</h4>", unsafe_allow_html=True)
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown(f"""
                                <div style='background: #132F4C; border: 1px solid #00D9FF; padding: 20px; border-radius: 15px; text-align: center;'>
                                    <div style='font-size: 2.5rem; font-weight: 700; margin-bottom: 5px; color: #00D9FF;'>{len(tool_calls)}/4</div>
                                    <div style='font-size: 0.9rem; color: #B2BAC2;'>Tools Used</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div style='background: #132F4C; border: 1px solid #00CCA3; padding: 20px; border-radius: 15px; text-align: center;'>
                                    <div style='font-size: 2.5rem; font-weight: 700; margin-bottom: 5px; color: #00CCA3;'>{elapsed_time:.0f}ms</div>
                                    <div style='font-size: 0.9rem; color: #B2BAC2;'>Processing Time</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                # Estimate cost based on tools used
                                cost_per_tool = 0.0005
                                total_cost = len(tool_calls) * cost_per_tool
                                st.markdown(f"""
                                <div style='background: #132F4C; border: 1px solid #FFD93D; padding: 20px; border-radius: 15px; text-align: center;'>
                                    <div style='font-size: 2.5rem; font-weight: 700; margin-bottom: 5px; color: #FFD93D;'>${total_cost:.4f}</div>
                                    <div style='font-size: 0.9rem; color: #B2BAC2;'>Estimated Cost</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Show raw messages for debugging
                            with st.expander("🔍 Debug: Raw Agent Messages"):
                                for i, msg in enumerate(messages):
                                    st.write(f"**Message {i+1}:** {type(msg).__name__}")
                                    st.write(f"Content: {getattr(msg, 'content', 'N/A')[:200]}")
                                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                        st.write(f"Tool calls: {len(msg.tool_calls)}")
                                    st.markdown("---")
                            
                        except Exception as e:
                            st.error(f"Error running agent: {e}")
                            import traceback
                            with st.expander("🔍 Error Details"):
                                st.code(traceback.format_exc())

# Example claims section
st.markdown("<br><br>", unsafe_allow_html=True)
with st.expander("💡 Example Healthcare Claims Library", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #00CCA3 0%, #00D9FF 100%); padding: 15px; border-radius: 10px; color: #0A1929; text-align: center; margin-bottom: 15px;'>
            <h4 style='margin: 0; color: #0A1929;'>✅ Legitimate Claims</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='sample-card'>
            <h5 style='color: #00D9FF; margin-top: 0;'>Legitimate Office Visit</h5>
        </div>
        """, unsafe_allow_html=True)
        st.code(SAMPLE_CLAIMS["Legitimate Office Visit"][:250] + "...", language="text")
        
        st.markdown("""
        <div class='sample-card'>
            <h5 style='color: #00D9FF; margin-top: 0;'>Legitimate Preventive Care</h5>
        </div>
        """, unsafe_allow_html=True)
        st.code(SAMPLE_CLAIMS["Legitimate Preventive Care"][:250] + "...", language="text")
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #FF6B6B 0%, #FF5252 100%); padding: 15px; border-radius: 10px; color: white; text-align: center; margin-bottom: 15px;'>
            <h4 style='margin: 0;'>🚨 Fraudulent Claims</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='sample-card'>
            <h5 style='color: #FF6B6B; margin-top: 0;'>Upcoding Scheme</h5>
        </div>
        """, unsafe_allow_html=True)
        st.code(SAMPLE_CLAIMS["Upcoding Scheme"][:250] + "...", language="text")
        
        st.markdown("""
        <div class='sample-card'>
            <h5 style='color: #FF6B6B; margin-top: 0;'>Phantom Billing</h5>
        </div>
        """, unsafe_allow_html=True)
        st.code(SAMPLE_CLAIMS["Phantom Billing"][:250] + "...", language="text")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 20px; background: #132F4C; border: 1px solid #1E4976; border-radius: 15px; margin-top: 40px;'>
    <p style='margin: 0; color: #00D9FF; font-weight: 600;'>
        🏥 Healthcare Payer Fraud Detection | Built with LangGraph + Unity Catalog AI Functions | ☁️ Databricks Apps
    </p>
</div>
""", unsafe_allow_html=True)
