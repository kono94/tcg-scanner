#!/bin/bash

# Set the input directory to the current working directory
INPUT_DIR=$(pwd)

# Loop through all .MOV files in the directory
for file in "$INPUT_DIR"/*.MOV; do
    # Check if the file exists to avoid running if there are no .MOV files
    if [ -e "$file" ]; then
        # Extract the filename without extension
        filename=$(basename -- "$file")
        filename_no_ext="${filename%.*}"

        # Define the output filename
        output="${filename_no_ext}.mp4"

        # Use ffmpeg to re-encode the file to .mp4
        ffmpeg -i "$file" -c:v libx264 -crf 23 -preset medium -c:a aac -b:a 128k "$INPUT_DIR/$output"

        echo "Converted $file to $output"
    else
        echo "No .MOV files found."
    fi
done
