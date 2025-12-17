#!/bin/bash
# Run Streamlit app locally for preview

echo "=========================================="
echo "🚀 Starting Streamlit App Locally"
echo "=========================================="
echo ""
echo "📝 Setting up environment variables..."

# Set environment variables from config.yaml
export CATALOG_NAME="ashraf"
export SCHEMA_NAME="claims_analysis"
export DATABRICKS_WAREHOUSE_ID="a94a22f8652d85c1"
export ENVIRONMENT="dev"
export LLM_ENDPOINT="databricks-claude-sonnet-4-5"

# Note: For full functionality, you'll need Databricks authentication
# The app will work for UI preview but won't connect to Databricks services
echo "⚠️  Note: Running in preview mode"
echo "    - UI and design will be fully functional"
echo "    - Databricks connections will not work without authentication"
echo ""
echo "🌐 Starting Streamlit on http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo ""

cd app
source venv/bin/activate
streamlit run app_databricks.py --server.port 8501 --server.address localhost

