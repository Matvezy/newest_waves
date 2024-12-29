#!/bin/bash

# Check for two arguments
if [ $# -ne 2 ]; then
    echo "Usage: $0 folder1 folder2"
    exit 1
fi

folder1="$1"
folder2="$2"

# Check if directories exist
if [ ! -d "$folder1" ]; then
    echo "Error: $folder1 is not a directory."
    exit 1
fi

if [ ! -d "$folder2" ]; then
    echo "Error: $folder2 is not a directory."
    exit 1
fi

# Count .jpg files in each folder
count1=$(find "$folder1" -type f -iname "*.jpg" | wc -l)
count2=$(find "$folder2" -type f -iname "*.jpg" | wc -l)

echo "Number of .jpg images in $folder1: $count1"
echo "Number of .jpg images in $folder2: $count2"
