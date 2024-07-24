#!/bin/bash

# Directory containing the annotations
ANNOTATIONS_DIR="./dataset/Annotations"

# Output file
OUTPUT_FILE="./dataset/train.txt"

# Check if the annotations directory exists
if [ ! -d "$ANNOTATIONS_DIR" ]; then
  echo "Annotations directory does not exist: $ANNOTATIONS_DIR"
  exit 1
fi

# Create or clear the output file
> "$OUTPUT_FILE"

# Read filenames, remove extensions, and write to output file
for filepath in "$ANNOTATIONS_DIR"/*.xml; do
  filename=$(basename "$filepath" .xml)
  echo "$filename" >> "$OUTPUT_FILE"
done

echo "train.txt has been created with the following content:"
cat "$OUTPUT_FILE"
