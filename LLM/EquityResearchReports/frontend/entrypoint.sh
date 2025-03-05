#!/bin/sh

# Check if ENVIRONMENT variable is set, default to 'dev'
ENVIRONMENT=${ENVIRONMENT:-dev}

# Run the replace_base_url.sh script
if [ -f /app/replace_base_url.sh ]; then
  echo "Running replace_base_url.sh script from entrypoint.sh..."
  /app/replace_base_url.sh
else
  echo "replace_base_url.sh script not found. Exiting."
  exit 1
fi

 # Start the server
  exec node /app/server.js
fi