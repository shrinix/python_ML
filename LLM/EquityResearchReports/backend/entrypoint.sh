#!/bin/sh

# Find the location of python3
PYTHON_PATH=$(which python3)

# Run the Python application
exec "$PYTHON_PATH" ER_chat_service.py