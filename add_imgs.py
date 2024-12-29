#!/usr/bin/env python3

import os
import sys
import shutil
from pathlib import Path
import argparse

def get_numeric_png_files(folder):
    """Returns a sorted list of numeric filenames from .png files in the folder."""
    files = []
    for file in os.listdir(folder):
        if file.lower().endswith('.png'):
            name = file[:-4]  # Remove .png extension
            if name.isdigit():
                files.append(name)
    return sorted(files, key=lambda x: int(x))

def main(folder1, folder2, dry_run=False, backup=False):
    folder1 = Path(folder1)
    folder2 = Path(folder2)

    # Check if directories exist
    if not folder1.is_dir():
        print(f"Error: {folder1} is not a directory.")
        sys.exit(1)
    if not folder2.is_dir():
        print(f"Error: {folder2} is not a directory.")
        sys.exit(1)

    # Optionally create backups
    if backup:
        backup_folder1 = folder1.parent / (folder1.name + '_backup')
        backup_folder2 = folder2.parent / (folder2.name + '_backup')
        if dry_run:
            print(f"Would create backup of {folder1} at {backup_folder1}")
            print(f"Would create backup of {folder2} at {backup_folder2}")
        else:
            print(f"Creating backup of {folder1} at {backup_folder1}")
            shutil.copytree(folder1, backup_folder1)
            print(f"Creating backup of {folder2} at {backup_folder2}")
            shutil.copytree(folder2, backup_folder2)

    # **Step 1:** Identify common numeric .png files in both folders
    print("\nStep 1: Identifying common numeric .png files in both folders...")
    files1 = get_numeric_png_files(folder1)
    files2 = get_numeric_png_files(folder2)
    common_files = sorted(set(files1) & set(files2), key=lambda x: int(x))

    if len(common_files) == 0:
        print("Error: No common images found between the two folders.")
        sys.exit(1)

    # **Step 2:** Define the number of images to duplicate and starting number
    number_of_images = int(input("Enter the number of images to duplicate: "))
    if len(common_files) < number_of_images:
        print(f"Error: Not enough common images to duplicate the last {number_of_images} images.")
        sys.exit(1)

    starting_number = int(input("Enter the starting number for new images (e.g., 1999): "))
    ending_number = starting_number + number_of_images - 1

    # **Step 3:** Get the last N images
    last_n_files = common_files[-number_of_images:]
    print(f"Last {number_of_images} common images: {[f + '.png' for f in last_n_files]}")

    # **Step 4:** Duplicate the last N images with new names starting from the specified number
    print(f"\nStep 2: Duplicating the last {number_of_images} images in both folders...")
    counter = starting_number

    for num_str in last_n_files:
        original_file1 = folder1 / f"{num_str}.png"
        original_file2 = folder2 / f"{num_str}.png"
        new_num_str = str(counter)
        new_file1 = folder1 / f"{new_num_str}.png"
        new_file2 = folder2 / f"{new_num_str}.png"

        # Check if the new filenames already exist to avoid overwriting
        if new_file1.exists() or new_file2.exists():
            print(f"Error: {new_file1} or {new_file2} already exists. Aborting to prevent overwriting.")
            sys.exit(1)

        if dry_run:
            print(f"Would copy {original_file1} to {new_file1}")
            print(f"Would copy {original_file2} to {new_file2}")
        else:
            shutil.copy2(original_file1, new_file1)
            shutil.copy2(original_file2, new_file2)
            print(f"Copied {original_file1} to {new_file1}")
            print(f"Copied {original_file2} to {new_file2}")

        counter += 1

    print("\nOperation complete. Both folders now have synchronized images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Duplicate the last N common images in two folders.")
    parser.add_argument('--folder1', default="/home/ubuntu/verb-workspace/WAVES/datasets/main/imagenet/tree_ring")
    parser.add_argument('--folder2', default="/home/ubuntu/verb-workspace/WAVES/datasets/attacked/imagenet/none-2-tree_ring")
    parser.add_argument('--dry-run', action='store_true', help="Run the script without making any changes.")
    parser.add_argument('--backup', action='store_true', help="Create backups before making changes.")

    args = parser.parse_args()

    main(args.folder1, args.folder2, dry_run=args.dry_run, backup=args.backup)
