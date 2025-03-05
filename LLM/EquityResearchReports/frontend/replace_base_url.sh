#!/bin/sh

# Check if BASE_URL variable is set
if [ -z "$BASE_URL" ]; then
  echo "BASE_URL is not set. Exiting."
  exit 1
fi

echo "Creating runtime configuration file with BASE_URL=$BASE_URL"

# Ensure the directory exists
mkdir -p dist/frontend/assets

# Create the runtime configuration file
cat <<EOF > dist/frontend/assets/runtime-config.json
{
  "BASE_URL": "$BASE_URL"
}
EOF