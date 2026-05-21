#!/bin/bash

export CLAUDE_CODE_USE_VERTEX=1
export CLOUD_ML_REGION=global
export ANTHROPIC_VERTEX_PROJECT_ID=ceo-proj-foxbrain-prod

# Optional: Override the Vertex endpoint URL for custom endpoints or gateways
# export ANTHROPIC_VERTEX_BASE_URL=https://aiplatform.googleapis.com

# Optional: Disable prompt caching if needed
# export DISABLE_PROMPT_CACHING=1

# Optional: Request 1-hour prompt cache TTL instead of the 5-minute default
# export ENABLE_PROMPT_CACHING_1H=1

# When CLOUD_ML_REGION=global, override region for models that don't support global endpoints
export VERTEX_REGION_CLAUDE_HAIKU_4_5=us-east5
export VERTEX_REGION_CLAUDE_4_6_SONNET=europe-west1

export ANTHROPIC_DEFAULT_OPUS_MODEL='claude-opus-4-7'
export ANTHROPIC_DEFAULT_SONNET_MODEL='claude-sonnet-4-6'
export ANTHROPIC_DEFAULT_HAIKU_MODEL='claude-haiku-4-5@20251001'

echo "Claude Code Chat"
echo "================"
echo "1) Resume last conversation"
echo "2) New conversation"
echo ""
read -p "Choose [1/2]: " choice

case $choice in
 1)
     IS_SANDBOX=1 claude --resume --dangerously-skip-permissions
     if [ $? -ne 0 ]; then
         echo "No previous conversation found. Starting new conversation..."
         IS_SANDBOX=1 claude --dangerously-skip-permissions
     fi
     ;;
 2)
     IS_SANDBOX=1 claude --dangerously-skip-permissions
     ;;
 *)
     echo "Invalid choice. Please enter 1 or 2."
     exit 1
     ;;
esac
