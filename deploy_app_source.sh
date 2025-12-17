#!/bin/bash
# Deploy App Source Code
# This script deploys the Streamlit app source code to Databricks Apps
#
# Usage:
#   ./deploy_app_source.sh [environment]
#
# Examples:
#   ./deploy_app_source.sh dev
#   ./deploy_app_source.sh staging
#   ./deploy_app_source.sh prod

set -e

# Skip TLS verification for macOS certificate issues
export DATABRICKS_INSECURE_TLS_SKIP_VERIFY=true
security find-certificate -a -p /Library/Keychains/System.keychain /System/Library/Keychains/SystemRootCertificates.keychain > ~/.databricks-certs.pem 2>/dev/null || true
export SSL_CERT_FILE=~/.databricks-certs.pem
export REQUESTS_CA_BUNDLE=~/.databricks-certs.pem

# Unset environment variables that might override profile settings
unset DATABRICKS_HOST
unset DATABRICKS_TOKEN

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

ENVIRONMENT=${1:-dev}

echo "========================================================================"
echo "🚀 DEPLOYING APP SOURCE CODE"
echo "========================================================================"
echo "Environment: ${ENVIRONMENT}"
echo ""

# Check if config.yaml exists
if [ ! -f "config.yaml" ]; then
    echo -e "${RED}❌ ERROR: config.yaml not found!${NC}"
    exit 1
fi

# Load config from config.yaml
echo "📝 Loading configuration..."
eval $(python3 -c "
import yaml
import sys
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    env = config['environments']['$ENVIRONMENT']
    print(f\"PROFILE='{env['profile']}'\")
    print(f\"APP_NAME='{env['app_name']}'\")
except Exception as e:
    print(f'ERROR: Failed to load config - {e}', file=sys.stderr)
    sys.exit(1)
")

if [ -z "$APP_NAME" ]; then
    echo -e "${RED}❌ ERROR: Failed to load configuration${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Configuration loaded${NC}"
echo "   Profile: $PROFILE"
echo "   App Name: $APP_NAME"
echo ""

# Check if app folder exists
if [ ! -d "./app" ]; then
    echo -e "${RED}❌ ERROR: ./app folder not found!${NC}"
    exit 1
fi

# Check if app.yaml exists
if [ ! -f "./app/app.yaml" ]; then
    echo -e "${RED}❌ ERROR: ./app/app.yaml not found!${NC}"
    echo -e "${YELLOW}💡 Run: python generate_app_yaml.py $ENVIRONMENT${NC}"
    exit 1
fi

echo "📦 Deploying app source code..."
echo "   Source: ./app"
echo "   Target: $APP_NAME"
echo ""

echo "📦 Deploying app source code..."
echo "   App: $APP_NAME"
echo ""

# Get current Databricks username for constructing bundle path
CURRENT_USER=$(databricks current-user me --profile "$PROFILE" --output json 2>/dev/null | python3 -c "import sys, json; print(json.load(sys.stdin)['userName'])" 2>&1)

if [ $? -ne 0 ] || [ -z "$CURRENT_USER" ]; then
    echo -e "${RED}❌ ERROR: Could not determine current Databricks user${NC}"
    echo "Please ensure Databricks CLI is configured properly"
    exit 1
fi

# Construct the bundle workspace path where source code is deployed
# Pattern: /Workspace/Users/<username>/.bundle/<bundle-name>/<environment>/files/app
BUNDLE_NAME="fraud_detection_claims"
BUNDLE_SOURCE_PATH="/Workspace/Users/${CURRENT_USER}/.bundle/${BUNDLE_NAME}/${ENVIRONMENT}/files/app"

echo "Current user: $CURRENT_USER"
echo "Bundle source path: $BUNDLE_SOURCE_PATH"
echo ""

# Deploy app from the bundle workspace path
# This is the standard pattern per Microsoft documentation
echo "Deploying app from bundle workspace location..."
databricks apps deploy "$APP_NAME" --source-code-path "$BUNDLE_SOURCE_PATH" --profile "$PROFILE"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ ERROR: App deployment failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ App source code deployed successfully!${NC}"
echo ""

# Check app status
echo "Checking app status..."
databricks apps get "$APP_NAME" --profile "$PROFILE" | grep -E "(state|default_source_code_path|url)" || true

echo ""
echo -e "${GREEN}✅ APP SOURCE CODE DEPLOYED SUCCESSFULLY!${NC}"
echo "========================================================================"
echo ""
echo "🔍 Checking app status..."
databricks apps get "$APP_NAME" --profile "$PROFILE" | grep -E "(state|url)" || true

echo ""
echo "🌐 App URL: https://frauddetection-dev-984752964297111.11.azure.databricksapps.com"
echo "   (or check: databricks apps get $APP_NAME --profile $PROFILE | grep url)"
echo ""
echo -e "${BLUE}💡 Next steps:${NC}"
echo "   1. Wait 30-60 seconds for app to start"
echo "   2. Open the app URL in your browser"
echo "   3. Check the app logs if you encounter any errors"
echo ""
echo "========================================================================"

