#!/bin/zsh

# Usage: ./convert-tabs-to-spaces.sh filename.yml

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <filename>"
  exit 1
fi

file="$1"

# Replace all tabs with 2 spaces (in-place)
expand -t 2 "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"

echo "Converted all tabs to spaces in $file"
