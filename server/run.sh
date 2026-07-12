#!/bin/bash
# Start R1 Voice Server
cd /home/mgt/projects/r1-voice-server

# Read API key from Hermes config
API_KEY=$(python3 -c "
import yaml
with open('/home/mgt/.hermes/config.yaml') as f:
    c = yaml.safe_load(f)
key = c.get('api_server', {}).get('api_key', '')
print(key if key else '')
" 2>/dev/null)

if [ -n "$API_KEY" ]; then
    export R1_HERMES_API_KEY="$API_KEY"
fi

exec .venv/bin/python server.py
