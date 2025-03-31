#!/bin/sh

# Check if ENVIRONMENT variable is set, default to 'dev'
ENVIRONMENT=${ENVIRONMENT:-dev}

# Run the replace_urls.sh script
if [ -f /app/replace_urls.sh ]; then
  echo "Running replace_urls.sh script from entrypoint.sh..."
  /app/replace_urls.sh
else
  echo "replace_urls.sh script not found. Exiting."
  exit 1
fi

 # Start the server
  exec node /app/server.js --replace-urls-executed=true
fi