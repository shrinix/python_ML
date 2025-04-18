#!/bin/sh
echo "Running entrypoint.sh"
# Check if an env-file is provided as a command line argument
# if [ "$1" = "--env-file" ] && [ -n "$2" ]; then
#   echo "Loading environment variables from $2"
#   if [ -f "$2" ]; then
#     export $(grep -v '^#' "$2" | xargs)
#   else
#     echo "Environment file $2 not found. Exiting."
#     # exit 1
#   fi
# fi

# Find the location of python3
PYTHON_PATH=$(which python3)

ls -lia /app

echo $PYTHON_PATH

$PYTHON_PATH --version

ls -lia /app/ER_metrics_service.py

env

# Run the Python application
exec "$PYTHON_PATH" /app/ER_metrics_service.py