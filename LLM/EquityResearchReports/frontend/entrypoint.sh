#!/bin/sh

# Check if ENVIRONMENT variable is set, default to 'dev'
ENVIRONMENT=${ENVIRONMENT:-dev}

# Copy the appropriate environment file
cp /app/environments/environment.${ENVIRONMENT}.ts /app/dist/frontend/assets/environment.ts

# Start the server
exec node /app/server.js