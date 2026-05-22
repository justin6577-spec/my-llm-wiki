#!/bin/bash
# setup_cron.sh — Install daily cron job for the wiki ingestion agent
VAULT=/work/HHRI-AI/Saqlain/my-wiki
CONDA=/work/HHRI-AI/anaconda/etc/profile.d/conda.sh
LOG=$VAULT/cron.log

# Build the cron command: source conda, activate env, then run agent
CRON_CMD="0 8 * * * source $CONDA && conda activate saqlain_vllm && cd $VAULT && python3 agent.py daily >> $LOG 2>&1"

# Append only if not already present
if crontab -l 2>/dev/null | grep -q "agent.py daily"; then
    echo "Cron job already exists — no change."
else
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
    echo "Cron job added: agent runs daily at 8:00 AM"
fi

echo ""
echo "Current crontab:"
crontab -l
