#!/bin/bash

DATA_DIR="ASP/data/faa_conll"

# Check if the 'data' directory already exists
if [ -d "$DATA_DIR" ]; then
    echo "Directory $DATA_DIR already exists."
else
    # If the directory does not exist, create it
    echo "Creating directory $DATA_DIR."
    mkdir "$DATA_DIR"
fi

# Paths for the source and destination files
source_file="../../data/FAA_data/faa.conll"
dest1="$DATA_DIR/dev.english.v4_gold_conll"
dest2="$DATA_DIR/train.english.v4_gold_conll"
dest3="$DATA_DIR/test.english.v4_gold_conll"

# Copy the file to the new location with the specified names
cp "$source_file" "$dest1"
echo "Copied to $dest1"
cp "$source_file" "$dest2"
echo "Copied to $dest2"
cp "$source_file" "$dest3"
echo "Copied to $dest3"