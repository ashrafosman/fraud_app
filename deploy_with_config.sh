#!/bin/bash
# Deploy script that auto-generates app.yaml

set -e  # Exit on error

# Skip TLS verification for macOS certificate issues
export DATABRICKS_INSECURE_TLS_SKIP_VERIFY=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get environment from argument or default to 'dev'
ENVIRONMENT=${1:-dev}

echo "========================================================================"
echo "🚀 FRAUD DETECTION DEPLOYMENT"
echo "========================================================================"
echo "Environment: ${ENVIRONMENT}"
echo ""

# Step 1: Validate config.yaml exists
if [ ! -f "config.yaml" ]; then
    echo -e "${RED}❌ ERROR: config.yaml not found!${NC}"
    echo "Please create config.yaml from config.yaml.template"
    exit 1
fi

# Step 2: Generate app.yaml from config.yaml
echo "📝 Step 1: Generating app/app.yaml from config.yaml..."
python generate_app_yaml.py ${ENVIRONMENT}

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ ERROR: Failed to generate app.yaml${NC}"
    exit 1
fi

echo -e "${GREEN}✅ app.yaml generated successfully${NC}"
echo ""

# Step 3: Validate databricks.yml exists
if [ ! -f "databricks.yml" ]; then
    echo -e "${RED}❌ ERROR: databricks.yml not found!${NC}"
    exit 1
fi

# Step 4: Deploy with Databricks Asset Bundles
echo "📦 Step 2: Deploying with Databricks Asset Bundles..."
databricks bundle deploy --target ${ENVIRONMENT} --profile work

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ ERROR: Deployment failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Deployment successful${NC}"
echo ""

# Step 5: Run setup job to create all resources
echo "⚙️  Step 3: Running setup job (creates catalog, tables, functions, data)..."
echo ""

databricks bundle run setup_fraud_detection --target ${ENVIRONMENT} --profile work

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ ERROR: Setup job failed${NC}"
    echo "Check the job logs in Databricks for details"
    exit 1
fi

echo -e "${GREEN}✅ Setup job completed successfully${NC}"
echo ""

# Step 6: Grant permissions to app service principal
echo "🔒 Step 4: Granting service principal permissions..."
echo ""
echo "⏳ Waiting 10 seconds for app to fully initialize..."
sleep 10

./grant_permissions.sh ${ENVIRONMENT}

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  WARNING: Permission grant failed${NC}"
    echo "You can manually grant permissions later by running:"
    echo "  ./grant_permissions.sh ${ENVIRONMENT}"
    echo ""
fi

# Step 7: Deploy app source code
echo "🚀 Step 5: Deploying app source code..."
echo ""
echo "⏳ Waiting for app to be ready for deployment (checking status)..."

# Wait a moment for the app to be fully initialized
sleep 5

# Check if there's an active deployment and wait for it
for i in {1..12}; do
    APP_STATUS=$(databricks apps get frauddetection-${ENVIRONMENT} --profile work --output json 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('compute_status', {}).get('state', 'UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")
    
    if [ "$APP_STATUS" != "DEPLOYING" ]; then
        echo "✅ App ready for deployment (status: $APP_STATUS)"
        break
    fi
    
    echo "  App is still deploying, waiting... ($i/12)"
    sleep 10
done

./deploy_app_source.sh ${ENVIRONMENT}

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  WARNING: App source deployment failed${NC}"
    echo "You can manually deploy app source later by running:"
    echo "  ./deploy_app_source.sh ${ENVIRONMENT}"
    echo ""
fi

echo "========================================================================"
echo -e "${GREEN}✅ DEPLOYMENT COMPLETE!${NC}"
echo "========================================================================"
echo ""
echo "What was deployed:"
echo "  ✅ Infrastructure (job definitions, app definition)"
echo "  ✅ Setup job executed (catalog, schema, tables, functions, data)"
echo "  ✅ Service principal permissions granted"
echo "  ✅ Streamlit app source code deployed"
echo ""
echo "⚠️  IMPORTANT: Configure Genie Space Instructions"
echo "  1. Go to Databricks: Data Intelligence > Genie > Fraud Detection Analytics"
echo "  2. Click Settings (gear icon)"
echo "  3. Copy instructions from setup job output (task: create_genie_space)"
echo "  4. Paste into Instructions field and Save"
echo ""
echo "Next steps:"
echo "  1. Wait 30-60 seconds for app to start"
echo "  2. Access app:"
echo "     https://<workspace>/apps/frauddetection-${ENVIRONMENT}"
echo ""
echo "Configuration used:"
echo "  - Environment: ${ENVIRONMENT}"
echo "  - Config file: config.yaml"
echo "  - Generated: app/app.yaml"
echo "========================================================================"


