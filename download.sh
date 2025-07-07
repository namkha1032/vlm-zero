#!/bin/bash

# Load .env if it exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Fail if HF_TOKEN is not set
if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN environment variable not set."
  echo "Please set it in your environment or in a .env file."
  exit 1
fi

mkdir -p weights

if [ ! -f links.txt ]; then
  echo "Error: links.txt not found."
  exit 1
fi

while IFS= read -r url; do
  [ -z "$url" ] && continue
  echo "Downloading $url into weights/"
  wget --header="Authorization: Bearer $HF_TOKEN" \
       --content-disposition \
       -P weights "$url"
done < links.txt
