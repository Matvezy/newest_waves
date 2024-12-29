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

# Delete all files up to 1999.jpg (not inclusive) in both folders
echo "Deleting files with names less than 1999.jpg in both folders..."

find "$folder1" -type f -name '[0-9]*.jpg' | while read -r file; do
    filename=$(basename "$file")
    filenum=${filename%.jpg}
    if [ "$filenum" -lt 1999 ]; then
        rm "$file"
        echo "Deleted $file"
    fi
done

find "$folder2" -type f -name '[0-9]*.jpg' | while read -r file; do
    filename=$(basename "$file")
    filenum=${filename%.jpg}
    if [ "$filenum" -lt 1999 ]; then
        rm "$file"
        echo "Deleted $file"
    fi
done

# Get list of files common to both folders (after deletion)
echo "Finding common files in both folders..."
common_files=($(comm -12 <(ls "$folder1" | sort) <(ls "$folder2" | sort)))

# Extract numerical filenames and sort them
common_numbers=($(printf '%s\n' "${common_files[@]}" | grep '^[0-9]\+\.jpg$' | sed 's/\.jpg$//' | sort -n))

# Get the last 64 images
echo "Selecting the last 64 images..."
total_common_files=${#common_numbers[@]}

if [ "$total_common_files" -lt 64 ]; then
    echo "Error: Not enough images to duplicate the last 64 images."
    exit 1
fi

start_index=$((total_common_files - 64))

last_64_numbers=("${common_numbers[@]:$start_index:64}")

# Duplicate the last 64 images in both folders with new names starting from 2000.jpg to 2063.jpg
echo "Duplicating the last 64 images..."

counter=2000
for num in "${last_64_numbers[@]}"; do
    original_file1="$folder1/$num.jpg"
    original_file2="$folder2/$num.jpg"
    new_file1="$folder1/$counter.jpg"
    new_file2="$folder2/$counter.jpg"

    # Copy and rename in folder1
    cp "$original_file1" "$new_file1"
    echo "Copied $original_file1 to $new_file1"

    # Copy and rename in folder2
    cp "$original_file2" "$new_file2"
    echo "Copied $original_file2 to $new_file2"

    ((counter++))
done

echo "Operation complete. Both folders now have synchronized images."
